#!/usr/bin/env bash
set -Eeuo pipefail

[[ "${P2_PREFETCH_CONFIRM_START:-}" == YES ]] || {
  echo "refusing GPU run: set P2_PREFETCH_CONFIRM_START=YES after reviewing the allocation and P0 contract" >&2
  exit 2
}

HARNESS_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CB=${P2_PREFETCH_CB:-/root/shared/.clusters/.tools/clusterbridge.sh}
LOCAL_REPO=${P2_PREFETCH_LOCAL_REPO:-/root/megascale_p2_prefetch}
P0_CONTRACT=${P2_PREFETCH_P0_CONTRACT:?set P2_PREFETCH_P0_CONTRACT to the validated P0 integration contract}
ALLOCATION=${P2_PREFETCH_ALLOCATION:?set P2_PREFETCH_ALLOCATION to one active 8xH20 allocation}
CALLER=${CODEX_THREAD_ID:-${CB_SID:-}}
CAMPAIGN=${P2_PREFETCH_CAMPAIGN_ID:-p2-prefetch-abba-$(date -u +%Y%m%dT%H%M%SZ)}
ROOT=${P2_PREFETCH_ARTIFACT_ROOT:-/root/shared/.clusters/.tmp/megascale_h3_p2_prefetch_abba}
DRY_RUN=${P2_PREFETCH_DRY_RUN:-0}
TEST_FAIL_AFTER_SNAPSHOT=${P2_PREFETCH_TEST_FAIL_AFTER_SNAPSHOT:-0}
POLL_S=${P2_PREFETCH_POLL_S:-60}
ARM_DEADLINE_S=${P2_PREFETCH_ARM_DEADLINE_S:-18000}
CB_TIMEOUT_S=${P2_PREFETCH_CB_TIMEOUT_S:-240}
CP_TIMEOUT_S=${P2_PREFETCH_CP_TIMEOUT_S:-1200}
COPY_FULL_REMOTE=${P2_PREFETCH_COPY_FULL_REMOTE:-0}

[[ "$CAMPAIGN" =~ ^p2-prefetch-abba-[A-Za-z0-9][A-Za-z0-9._-]{5,100}$ ]] || {
  echo "unsafe campaign ID" >&2
  exit 2
}
[[ "$CALLER" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] || {
  echo "cannot establish caller session UUID" >&2
  exit 2
}
[[ "$DRY_RUN" =~ ^[01]$ && "$COPY_FULL_REMOTE" =~ ^[01]$ ]] || exit 2
[[ "$TEST_FAIL_AFTER_SNAPSHOT" =~ ^[01]$ ]] || exit 2
[[ "$TEST_FAIL_AFTER_SNAPSHOT" == 0 || "$DRY_RUN" == 1 ]] || {
  echo "P2_PREFETCH_TEST_FAIL_AFTER_SNAPSHOT is allowed only in dry-run mode" >&2
  exit 2
}
[[ "$POLL_S" =~ ^[1-9][0-9]*$ && "$ARM_DEADLINE_S" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ -f "$P0_CONTRACT" && -x "$CB" && -d "$LOCAL_REPO" ]] || {
  echo "missing P0 contract, cb, or local repo" >&2
  exit 2
}

ART="$ROOT/results/$CAMPAIGN"
REMOTE_ROOT="/root/.cache/megascale_h3_p2_prefetch_${CAMPAIGN}"
[[ ! -e "$ART" ]] || {
  echo "refusing existing artifact directory: $ART" >&2
  exit 73
}

cb() {
  timeout --signal=TERM --kill-after=30s "${CB_TIMEOUT_S}s" env CB_SID="$CALLER" "$CB" "$@"
}

allocation_json=$(mktemp)
NODE=
CURRENT_RUN=
NVML_ACTIVE=0
CAMPAIGN_PREPARED=0

write_partial_summary() {
  local exit_code=$1 line=$2 command=$3
  [[ -d "$ART" ]] || return 0
  local writer="$HARNESS_ROOT/partial_summary.py"
  [[ -f "${HARNESS:-}/partial_summary.py" ]] && writer="$HARNESS/partial_summary.py"
  local args=(
    --artifact "$ART"
    --campaign-id "$CAMPAIGN"
    --exit-code "$exit_code"
    --failed-line "$line"
    --failed-command "$command"
  )
  [[ -n "$CURRENT_RUN" ]] && args+=(--current-run "$CURRENT_RUN")
  python3 "$writer" "${args[@]}" >/dev/null 2>&1 || true
}

stop_current_nvml_best_effort() {
  [[ "$NVML_ACTIVE" == 1 && -n "$CURRENT_RUN" && -n "$NODE" ]] || return 0
  cb "$NODE" run "set +e
    pidf='$REMOTE_ROOT/nvml_${CURRENT_RUN}.pid'
    if test -f \"\$pidf\"; then
      pid=\$(cat \"\$pidf\")
      test -d /proc/\$pid && kill \$pid
    fi
    exit 0" >/dev/null 2>&1 || true
  NVML_ACTIVE=0
}

kill_current_workload_best_effort() {
  [[ -n "$CURRENT_RUN" && -n "$NODE" ]] || return 0
  cb "$NODE" run "set +e
    identity='$REMOTE_ROOT/${CURRENT_RUN}.identity'
    expected='p2-prefetch-${CAMPAIGN}-${CURRENT_RUN}'
    if test -f \"\$identity\"; then
      pid=\$(awk -F= '/^pid=/{print \$2}' \"\$identity\")
      token=\$(awk -F= '/^token=/{print \$2}' \"\$identity\")
      start=\$(awk -F= '/^starttime=/{print \$2}' \"\$identity\")
      if test \"\$token\" = \"\$expected\" && test -r \"/proc/\$pid/stat\"; then
        observed=\$(awk '{print \$22}' \"/proc/\$pid/stat\")
        if test -n \"\$start\" && test \"\$observed\" = \"\$start\"; then
          kill -INT -- -\"\$pid\" 2>/dev/null || kill -INT \"\$pid\" 2>/dev/null || true
          sleep 5
          test -d \"/proc/\$pid\" && kill -KILL -- -\"\$pid\" 2>/dev/null || true
        fi
      fi
    fi
    exit 0" >/dev/null 2>&1 || true
}

best_effort_collect_partial() {
  [[ -n "$NODE" && -d "$ART" ]] || return 0
  mkdir -p "$ART/partial-remote"
  local names=(source-prepared.json .campaign-owner)
  if [[ -n "$CURRENT_RUN" ]]; then
    names+=(
      "${CURRENT_RUN}.log"
      "${CURRENT_RUN}.status"
      "${CURRENT_RUN}.started"
      "${CURRENT_RUN}.identity"
      "${CURRENT_RUN}.contract.json"
      "${CURRENT_RUN}.receipt.json"
      "sample_ledger_${CURRENT_RUN}.jsonl"
      "update_membership_${CURRENT_RUN}.json"
      "frozen_checkpoint_loaded_${CURRENT_RUN}.json"
      "runtime_summary_${CURRENT_RUN}.json"
      "nvml_${CURRENT_RUN}_${NODE}.csv"
      "nvml_${CURRENT_RUN}_${NODE}.stderr"
      "source-post-${CURRENT_RUN}.json"
    )
  fi
  local name
  for name in "${names[@]}"; do
    cb "$NODE" run "test -f '$REMOTE_ROOT/$name'" >/dev/null 2>&1 || continue
    cb cp "$NODE:$REMOTE_ROOT/$name" "$ART/partial-remote/$name" >/dev/null 2>&1 || true
  done
}

cleanup() {
  local exit_code=$? line=${BASH_LINENO[0]:-0} command=${BASH_COMMAND:-unknown}
  trap - EXIT INT TERM ERR
  if ((exit_code != 0)); then
    stop_current_nvml_best_effort
    kill_current_workload_best_effort
    best_effort_collect_partial
    write_partial_summary "$exit_code" "$line" "$command"
  fi
  if [[ "$CAMPAIGN_PREPARED" == 1 && -n "$NODE" ]]; then
    cb "$NODE" run "bash /root/shared/.clusters/.tools/node_cleanup.sh" >/dev/null 2>&1 || true
  fi
  rm -f "$allocation_json"
  exit "$exit_code"
}
trap cleanup EXIT INT TERM ERR

cb resource-status "$ALLOCATION" --json >"$allocation_json"
mapfile -t allocation_values < <(python3 - "$allocation_json" "$ALLOCATION" "$CALLER" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
expected, owner = sys.argv[2:]
allocation = payload.get("allocation") or {}
checks = {
    "ok": payload.get("ok") is True,
    "id": allocation.get("id") == expected,
    "owner": allocation.get("owner") == owner,
    "status": allocation.get("status") == "active",
    "kind": allocation.get("resource_kind") == "gpu",
    "count": allocation.get("gpu_count") == 8,
    "nodes": isinstance(allocation.get("nodes"), list) and len(allocation["nodes"]) == 1,
}
details = {item.get("ip"): item for item in allocation.get("node_details") or []}
node = (allocation.get("nodes") or [None])[0]
detail = details.get(node, {})
checks.update(
    {
        "gpus": detail.get("gpus") == 8,
        "type": detail.get("gpu_type") == "H20",
        "alive": detail.get("alive") is True,
        "fresh": detail.get("fresh") is True,
        "healthy": detail.get("healthy") is True,
        "idle": detail.get("real_workload") is False,
    }
)
bad = [key for key, passed in checks.items() if not passed]
if bad:
    raise SystemExit("allocation gate failed: " + ",".join(bad))
print(node)
PY
)
[[ ${#allocation_values[@]} -eq 1 ]] || exit 2
NODE=${allocation_values[0]}

mkdir -p "$ART/harness"
for file in \
  harness_lib.py render_plan.py preflight.py seal_source.py verify_source_bundle.py \
  snapshot_contract.py prepare_runtime_overlay.py aggregate_runtime_events.py partial_summary.py \
  launch_arm.sh run_abba.sh \
  summarize.py extract_offline_wandb.py collect_arm.py protocol_harness.py README.md; do
  [[ -f "$HARNESS_ROOT/$file" ]] || {
    echo "missing harness file: $file" >&2
    exit 2
  }
  cp --reflink=auto "$HARNESS_ROOT/$file" "$ART/harness/$file"
done
mkdir -p "$ART/harness/runtime_overlay"
for file in p2_runtime_receipts.py sitecustomize.py; do
  [[ -f "$HARNESS_ROOT/runtime_overlay/$file" ]] || {
    echo "missing runtime overlay file: $file" >&2
    exit 2
  }
  cp --reflink=auto \
    "$HARNESS_ROOT/runtime_overlay/$file" \
    "$ART/harness/runtime_overlay/$file"
done
python3 "$ART/harness/snapshot_contract.py" \
  --source "$P0_CONTRACT" \
  --output "$ART/p0_contract.json" \
  --input-dir "$ART/contract-inputs" \
  --receipt "$ART/contract-snapshot.json"
chmod -R a-w "$ART/harness"
HARNESS="$ART/harness"
python3 "$HARNESS/render_plan.py" \
  --campaign-id "$CAMPAIGN" \
  --node "$NODE" \
  --p0-contract "$ART/p0_contract.json" \
  --output "$ART/plan.json"
python3 "$HARNESS/preflight.py" \
  --repo "$LOCAL_REPO" \
  --p0-contract "$ART/p0_contract.json" \
  --plan "$ART/plan.json" \
  --campaign-id "$CAMPAIGN" \
  --allow-harness-changes \
  --output "$ART/preflight_local.json"
python3 - "$ART" "$CAMPAIGN" "$ALLOCATION" "$NODE" <<'PY'
import hashlib
import json
import pathlib
import sys
import time

artifact = pathlib.Path(sys.argv[1])
campaign, allocation, node = sys.argv[2:]


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


hashes = {
    path.relative_to(artifact / "harness").as_posix(): sha(path)
    for path in sorted((artifact / "harness").rglob("*"))
    if path.is_file()
}
payload = {
    "schema": "unirl-minimax-h3-p2-prefetch-abba-manifest-v1",
    "campaign_id": campaign,
    "allocation_id": allocation,
    "node": node,
    "plan_sha256": sha(artifact / "plan.json"),
    "p0_contract_sha256": sha(artifact / "p0_contract.json"),
    "harness_sha256": hashes,
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
(artifact / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

if [[ "$TEST_FAIL_AFTER_SNAPSHOT" == 1 ]]; then
  echo "intentional CPU-only protocol failure after immutable snapshot" >&2
  false
fi

if [[ "$DRY_RUN" == 1 ]]; then
  echo "P2_PREFETCH_DRY_RUN_COMPLETE ART=$ART ALLOCATION=$ALLOCATION NODE=$NODE"
  exit 0
fi

cb "$NODE" run "set -euo pipefail
  bash /root/shared/.clusters/.tools/node_cleanup.sh
  test ! -e '$REMOTE_ROOT'
  mkdir -m 0700 '$REMOTE_ROOT'
  printf '%s\n' '$CAMPAIGN' >'$REMOTE_ROOT/.campaign-owner'
  python3 '$HARNESS/verify_source_bundle.py' --p0-contract '$ART/p0_contract.json' --extract-to '$REMOTE_ROOT/source-integration' --output '$REMOTE_ROOT/source-prepared.json'
  python3 '$HARNESS/prepare_runtime_overlay.py' --p0-contract '$ART/p0_contract.json' --plan '$ART/plan.json' --destination '$REMOTE_ROOT/runtime-overlay' --output '$REMOTE_ROOT/runtime-overlay-prepared.json'
  cp '$HARNESS/aggregate_runtime_events.py' '$HARNESS/harness_lib.py' '$REMOTE_ROOT/runtime-overlay/'
  chmod -R a-w '$REMOTE_ROOT/runtime-overlay'
  test -d /root/.cache/megascale_h3/model
  test -x /root/unirl/.venv/bin/python
  test \"\$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)\" -eq 8
  test \"\$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | tr -d ' ')\" = NVIDIAH20"
CAMPAIGN_PREPARED=1
cb cp "$NODE:$REMOTE_ROOT/source-prepared.json" "$ART/source-prepared.json"

start_nvml() {
  local run=$1
  cb "$NODE" run "set -euo pipefail
    out='$REMOTE_ROOT/nvml_${run}_${NODE}.csv'
    err='$REMOTE_ROOT/nvml_${run}_${NODE}.stderr'
    pidf='$REMOTE_ROOT/nvml_${run}.pid'
    nohup bash -c 'exec -a nvml-$CAMPAIGN-$run nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,utilization.memory,pstate --format=csv,noheader,nounits -lms 500' >\"\$out\" 2>\"\$err\" </dev/null &
    echo \$! >\"\$pidf\""
  NVML_ACTIVE=1
}

stop_nvml() {
  local run=$1
  cb "$NODE" run "set -euo pipefail
    pid=\$(cat '$REMOTE_ROOT/nvml_${run}.pid')
    test -d /proc/\$pid && kill \$pid || true
    sleep 1
    test -s '$REMOTE_ROOT/nvml_${run}_${NODE}.csv'"
  NVML_ACTIVE=0
}

wait_arm() {
  local run=$1 elapsed=0
  while ((elapsed < ARM_DEADLINE_S)); do
    if cb "$NODE" run "test -f '$REMOTE_ROOT/${run}.status'" >/dev/null 2>&1; then
      cb "$NODE" run "cat '$REMOTE_ROOT/${run}.status'"
      return 0
    fi
    sleep "$POLL_S"
    elapsed=$((elapsed + POLL_S))
  done
  echo "arm timed out: $run" >&2
  return 2
}

for period in 1 2 3 4; do
  CURRENT_RUN=$(python3 - "$ART/plan.json" "$period" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1]))
print(next(arm["run_name"] for arm in plan["arms"] if arm["period"] == int(sys.argv[2])))
PY
)
  cb "$NODE" run "bash /root/shared/.clusters/.tools/node_cleanup.sh"
  start_nvml "$CURRENT_RUN"
  cb "$NODE" run \
    "P2_PREFETCH_CONFIRM_START=YES bash '$HARNESS/launch_arm.sh' '$ART/plan.json' '$period' '$REMOTE_ROOT'"
  wait_arm "$CURRENT_RUN"
  stop_nvml "$CURRENT_RUN"
  cb "$NODE" run "set -euo pipefail
    status=\$(awk -F= '/^exit_code=/{print \$2}' '$REMOTE_ROOT/${CURRENT_RUN}.status')
    test \"\$status\" = 0
    python3 '$HARNESS/verify_source_bundle.py' --p0-contract '$ART/p0_contract.json' --verify-dir '$REMOTE_ROOT/source-integration' --output '$REMOTE_ROOT/source-post-${CURRENT_RUN}.json'"
  cb cp "$NODE:$REMOTE_ROOT/source-post-${CURRENT_RUN}.json" "$ART/source-post-${CURRENT_RUN}.json"

  collect_root="$ART/.collect-${CURRENT_RUN}"
  mkdir "$collect_root"
  for name in \
    "${CURRENT_RUN}.log" "${CURRENT_RUN}.status" "${CURRENT_RUN}.started" \
    "${CURRENT_RUN}.identity" "${CURRENT_RUN}.contract.json" "${CURRENT_RUN}.receipt.json" \
    "sample_ledger_${CURRENT_RUN}.jsonl" "update_membership_${CURRENT_RUN}.json" \
    "frozen_checkpoint_loaded_${CURRENT_RUN}.json" "runtime_summary_${CURRENT_RUN}.json" \
    "nvml_${CURRENT_RUN}_${NODE}.csv" "nvml_${CURRENT_RUN}_${NODE}.stderr"; do
    cb cp "$NODE:$REMOTE_ROOT/$name" "$collect_root/$name"
  done
  cb cp \
    "$NODE:$REMOTE_ROOT/runtime_receipts_${CURRENT_RUN}" \
    "$collect_root/runtime_receipts_${CURRENT_RUN}"
  mkdir -p "$collect_root/hydra/$CURRENT_RUN/.hydra" "$collect_root/wandb"
  cb cp \
    "$NODE:$REMOTE_ROOT/hydra/$CURRENT_RUN/.hydra/config.yaml" \
    "$collect_root/hydra/$CURRENT_RUN/.hydra/config.yaml"
  cb cp "$NODE:$REMOTE_ROOT/wandb/$CURRENT_RUN" "$collect_root/wandb/$CURRENT_RUN"
  python3 "$HARNESS/collect_arm.py" \
    --remote-root "$collect_root" \
    --artifact "$ART" \
    --run-name "$CURRENT_RUN" \
    --node "$NODE" \
    --extractor "$HARNESS/extract_offline_wandb.py"
  cp -a \
    "$collect_root/runtime_receipts_${CURRENT_RUN}" \
    "$ART/runtime_receipts_${CURRENT_RUN}"
  if [[ "$COPY_FULL_REMOTE" == 1 ]]; then
    cb sync \
      "$NODE:$REMOTE_ROOT/" \
      "$ART/full_remote_${CURRENT_RUN}/" \
      --delete \
      --io-timeout 120 \
      --total-timeout "$CP_TIMEOUT_S"
  fi
  rm -rf "$collect_root"
  CURRENT_RUN=
done

python3 "$HARNESS/summarize.py" \
  --artifact "$ART" \
  --campaign-id "$CAMPAIGN" \
  --p0-contract "$ART/p0_contract.json" \
  --output "$ART/summary.json"
cb "$NODE" run "bash /root/shared/.clusters/.tools/node_cleanup.sh"
CAMPAIGN_PREPARED=0
echo "P2_PREFETCH_ABBA_COMPLETE ART=$ART SUMMARY=$ART/summary.json"
