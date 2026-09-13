#!/usr/bin/env bash
set -Eeuo pipefail

[[ "${P2_PREFETCH_CONFIRM_START:-}" == YES ]] || {
  echo "refusing launch: set P2_PREFETCH_CONFIRM_START=YES only after allocation/preflight review" >&2
  exit 2
}

plan=${1:?plan.json}
period=${2:?period 1..4}
remote_root=${3:?unique remote campaign root}
repo="$remote_root/source-integration"
python_env=${P2_PREFETCH_PYTHON_ENV:-/root/unirl/.venv}
model=${P2_PREFETCH_MODEL:-/root/.cache/megascale_h3/model}
base_site=${P2_PREFETCH_BASE_SITE:-/root/.cache/megascale_h3/venv/lib/python3.12/site-packages}
timeout_s=${P2_PREFETCH_ARM_TIMEOUT_S:-14400}

[[ "$period" =~ ^[1-4]$ ]] || { echo "period must be 1..4" >&2; exit 2; }
[[ "$remote_root" == /root/.cache/megascale_h3_p2_prefetch_* ]] || {
  echo "refusing unsafe remote root: $remote_root" >&2
  exit 2
}
[[ -f "$plan" && -d "$repo" && -x "$python_env/bin/python" && -d "$model" ]] || {
  echo "missing plan/prepared integration source/python/model" >&2
  exit 2
}
[[ -f "$remote_root/.campaign-owner" ]] || {
  echo "remote campaign root not prepared" >&2
  exit 2
}

mapfile -t values < <("$python_env/bin/python" - "$plan" "$period" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1]))
period = int(sys.argv[2])
arms = [row for row in plan.get("arms", []) if row.get("period") == period]
if len(arms) != 1:
    raise SystemExit("plan does not contain exactly one requested period")
arm = arms[0]
for value in (
    plan["campaign_id"],
    arm["run_name"],
    arm["treatment"],
    str(arm["prefetch"]).lower(),
    plan["p0_contract"]["path"],
    plan["p0_contract"]["sha256"],
    plan["prompt_manifest"]["path"],
    plan["prompt_manifest"]["sha256"],
    plan["frozen_checkpoint"]["load_dir"],
    plan["frozen_checkpoint"]["audit_sha256"],
    plan["frozen_checkpoint"]["tree_sha256"],
    plan["frozen_checkpoint"]["model_summary_sha256"],
    plan["frozen_checkpoint"]["adapter_key_signature_sha256"],
    plan["planner_target"],
    plan["planner"]["contract_sha256"],
    plan["two_update"]["contract_sha256"],
    plan["source_commit"],
    plan["source_tree"],
    plan["source_parent_commit"],
    plan["source_parent_tree"],
    plan["source_patch_sha256"],
    plan["integration_source"]["commit"],
    plan["integration_source"]["tree"],
    plan["integration_source"]["full_tree_sha256"],
    plan["runtime_overlay"]["base_module_sha256"],
):
    print(value)
PY
)
[[ ${#values[@]} -eq 25 ]] || { echo "failed to read arm plan" >&2; exit 2; }
campaign=${values[0]}
run_name=${values[1]}
treatment=${values[2]}
prefetch=${values[3]}
p0_contract_path=${values[4]}
p0_sha=${values[5]}
prompt_manifest=${values[6]}
prompt_sha=${values[7]}
load_dir=${values[8]}
checkpoint_audit_sha=${values[9]}
checkpoint_tree_sha=${values[10]}
checkpoint_model_sha=${values[11]}
checkpoint_adapter_sha=${values[12]}
planner_target=${values[13]}
planner_contract_sha=${values[14]}
two_update_contract_sha=${values[15]}
p2_commit=${values[16]}
p2_tree=${values[17]}
p2_parent_commit=${values[18]}
p2_parent_tree=${values[19]}
p2_patch_sha=${values[20]}
integration_commit=${values[21]}
integration_tree=${values[22]}
integration_full_tree=${values[23]}
base_overlay_sha=${values[24]}

[[ "$(cat "$remote_root/.campaign-owner")" == "$campaign" ]] || {
  echo "campaign owner mismatch" >&2
  exit 2
}
[[ "$treatment" == off && "$prefetch" == false || "$treatment" == on && "$prefetch" == true ]] || exit 2
[[ "$planner_target" == unirl.train.stack.GroupInterleavedCountPlanner ]] || exit 2
[[ "$(sha256sum "$prompt_manifest" | awk '{print $1}')" == "$prompt_sha" ]] || {
  echo "prompt manifest changed" >&2
  exit 2
}
source_identity="$repo/.p2-source-identity.json"
[[ -f "$source_identity" ]] || { echo "prepared source identity is missing" >&2; exit 2; }
"$python_env/bin/python" - "$source_identity" "$integration_commit" "$integration_tree" "$integration_full_tree" "$p0_sha" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
checks = {
    "completed": payload.get("completed") is True,
    "head": payload.get("head") == sys.argv[2],
    "tree": payload.get("tree") == sys.argv[3],
    "full_tree": payload.get("full_tree_sha256") == sys.argv[4],
    "p0": payload.get("p0_contract_sha256") == sys.argv[5],
}
bad = [key for key, value in checks.items() if not value]
if bad:
    raise SystemExit("prepared source identity mismatch: " + ",".join(bad))
PY

for value in \
  "$planner_contract_sha" "$two_update_contract_sha" "$checkpoint_audit_sha" \
  "$checkpoint_tree_sha" "$checkpoint_model_sha" "$checkpoint_adapter_sha" \
  "$p2_patch_sha" "$base_overlay_sha"; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || exit 2
done

log="$remote_root/${run_name}.log"
status="$remote_root/${run_name}.status"
started="$remote_root/${run_name}.started"
identity="$remote_root/${run_name}.identity"
contract="$remote_root/${run_name}.contract.json"
receipt="$remote_root/${run_name}.receipt.json"
hydra_dir="$remote_root/hydra/$run_name"
wandb_dir="$remote_root/wandb/$run_name"
runtime_receipts="$remote_root/runtime_receipts_${run_name}"
ledger="$remote_root/sample_ledger_${run_name}.jsonl"
updates="$remote_root/update_membership_${run_name}.json"
checkpoint_loaded="$remote_root/frozen_checkpoint_loaded_${run_name}.json"
runtime_summary="$remote_root/runtime_summary_${run_name}.json"
for path in \
  "$log" "$status" "$started" "$identity" "$contract" "$receipt" \
  "$hydra_dir" "$wandb_dir" "$runtime_receipts" "$ledger" "$updates" \
  "$checkpoint_loaded" "$runtime_summary"; do
  [[ ! -e "$path" ]] || { echo "refusing overwrite: $path" >&2; exit 73; }
done
mkdir -p "$hydra_dir" "$wandb_dir"

"$python_env/bin/python" - \
  "$contract" "$plan" "$period" "$p0_sha" "$prompt_sha" "$load_dir" \
  "$checkpoint_audit_sha" "$checkpoint_tree_sha" "$checkpoint_model_sha" \
  "$checkpoint_adapter_sha" "$planner_contract_sha" "$two_update_contract_sha" <<'PY'
import json
import os
import sys
import time

(
    out,
    plan_path,
    period,
    p0_sha,
    prompt_sha,
    load_dir,
    checkpoint_audit_sha,
    checkpoint_tree_sha,
    checkpoint_model_sha,
    checkpoint_adapter_sha,
    planner_contract_sha,
    two_update_contract_sha,
) = sys.argv[1:]
plan = json.load(open(plan_path))
arm = next(row for row in plan["arms"] if row["period"] == int(period))
payload = {
    "schema": "unirl-minimax-h3-p2-prefetch-arm-contract-v1",
    "campaign_id": plan["campaign_id"],
    "period": int(period),
    "arm_id": arm["arm_id"],
    "run_name": arm["run_name"],
    "treatment": arm["treatment"],
    "prefetch": arm["prefetch"],
    "p2_implementation_commit": plan["source_commit"],
    "p2_implementation_tree": plan["source_tree"],
    "p2_parent_commit": plan["source_parent_commit"],
    "p2_parent_tree": plan["source_parent_tree"],
    "p2_patch_sha256": plan["source_patch_sha256"],
    "integration_source_commit": plan["integration_source"]["commit"],
    "integration_source_tree": plan["integration_source"]["tree"],
    "integration_full_tree_sha256": plan["integration_source"]["full_tree_sha256"],
    "p0_contract_sha256": p0_sha,
    "prompt_manifest_sha256": prompt_sha,
    "frozen_checkpoint_load_dir": load_dir,
    "frozen_checkpoint_audit_sha256": checkpoint_audit_sha,
    "frozen_checkpoint_tree_sha256": checkpoint_tree_sha,
    "frozen_checkpoint_model_summary_sha256": checkpoint_model_sha,
    "frozen_checkpoint_adapter_key_signature_sha256": checkpoint_adapter_sha,
    "planner_contract_sha256": planner_contract_sha,
    "two_update_contract_sha256": two_update_contract_sha,
    "overrides": arm["overrides"],
    "required_runtime_artifacts": {
        "runtime_events": f"runtime_receipts_{arm['run_name']}/events",
        "runtime_summary": f"runtime_summary_{arm['run_name']}.json",
        "sample_ledger": f"sample_ledger_{arm['run_name']}.jsonl",
        "update_membership": f"update_membership_{arm['run_name']}.json",
        "frozen_checkpoint_loaded": f"frozen_checkpoint_loaded_{arm['run_name']}.json",
    },
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
temporary = f"{out}.tmp.{os.getpid()}"
open(temporary, "w").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, out)
PY
contract_sha=$(sha256sum "$contract" | awk '{print $1}')
printf 'started_utc=%s\ncampaign=%s\nrun=%s\nperiod=%s\ntreatment=%s\nprefetch=%s\n' \
  "$(date -u +%FT%TZ)" "$campaign" "$run_name" "$period" "$treatment" "$prefetch" >"$started"

cd "$repo"
nohup setsid bash -c '
  set +e
  status=$1
  receipt=$2
  identity=$3
  started=$4
  contract=$5
  campaign=$6
  run=$7
  period=$8
  treatment=$9
  p0_sha=${10}
  prompt_sha=${11}
  checkpoint_audit_sha=${12}
  checkpoint_tree_sha=${13}
  checkpoint_model_sha=${14}
  checkpoint_adapter_sha=${15}
  planner_contract_sha=${16}
  two_update_contract_sha=${17}
  runtime_summary=${18}
  ledger=${19}
  updates=${20}
  checkpoint_loaded=${21}
  shift 21
  for _ in $(seq 1 200); do
    test -s "$identity" && break
    sleep 0.05
  done
  test -s "$identity" || exit_code=125
  if test -z "${exit_code:-}"; then
    "$@"
    exit_code=$?
  fi
  status_tmp="${status}.tmp.$$"
  printf "exit_code=%s\nfinished_utc=%s\n" "$exit_code" "$(date -u +%FT%TZ)" >"$status_tmp"
  receipt_tmp="${receipt}.tmp.$$"
  python3 - \
    "$receipt_tmp" "$campaign" "$run" "$period" "$treatment" "$p0_sha" "$prompt_sha" \
    "$checkpoint_audit_sha" "$checkpoint_tree_sha" "$checkpoint_model_sha" "$checkpoint_adapter_sha" \
    "$planner_contract_sha" "$two_update_contract_sha" "$exit_code" "$contract" "$identity" "$started" \
    "$status_tmp" "$runtime_summary" "$ledger" "$updates" "$checkpoint_loaded" <<'"'"'PY'"'"'
import hashlib
import json
import os
import sys
import time

(
    out,
    campaign,
    run,
    period,
    treatment,
    p0,
    prompt,
    checkpoint_audit,
    checkpoint_tree,
    checkpoint_model,
    checkpoint_adapter,
    planner_contract,
    two_update_contract,
    rc,
    contract,
    identity,
    started,
    status,
    runtime_summary,
    ledger,
    updates,
    checkpoint_loaded,
) = sys.argv[1:]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.isfile(path) else None


payload = {
    "schema": "unirl-minimax-h3-p2-prefetch-arm-receipt-v1",
    "campaign_id": campaign,
    "run_name": run,
    "period": int(period),
    "treatment": treatment,
    "p0_contract_sha256": p0,
    "prompt_manifest_sha256": prompt,
    "frozen_checkpoint_audit_sha256": checkpoint_audit,
    "frozen_checkpoint_tree_sha256": checkpoint_tree,
    "frozen_checkpoint_model_summary_sha256": checkpoint_model,
    "frozen_checkpoint_adapter_key_signature_sha256": checkpoint_adapter,
    "planner_contract_sha256": planner_contract,
    "two_update_contract_sha256": two_update_contract,
    "exit_code": int(rc),
    "contract_sha256": sha(contract),
    "identity_sha256": sha(identity),
    "started_sha256": sha(started),
    "status_sha256": sha(status),
    "runtime_summary_sha256": sha(runtime_summary),
    "sample_ledger_sha256": sha(ledger),
    "update_membership_sha256": sha(updates),
    "frozen_checkpoint_loaded_sha256": sha(checkpoint_loaded),
    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
open(out, "w").write(json.dumps(payload, sort_keys=True) + "\n")
PY
  mv "$receipt_tmp" "$receipt"
  mv "$status_tmp" "$status"
  exit "$exit_code"
' "p2-prefetch-${campaign}-${run_name}" \
  "$status" "$receipt" "$identity" "$started" "$contract" "$campaign" "$run_name" \
  "$period" "$treatment" "$p0_sha" "$prompt_sha" "$checkpoint_audit_sha" \
  "$checkpoint_tree_sha" "$checkpoint_model_sha" "$checkpoint_adapter_sha" \
  "$planner_contract_sha" "$two_update_contract_sha" "$runtime_summary" "$ledger" \
  "$updates" "$checkpoint_loaded" \
  env \
  PYTHONPATH="$remote_root/runtime-overlay:$repo:$base_site" \
  PRETRAINED_MODEL="$model" VENV_DIR="$python_env" INSTALL_EDITABLE=0 \
  GPUS_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline WANDB_SILENT=true \
  WANDB_DIR="$wandb_dir" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
  TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HYDRA_FULL_ERROR=1 RAY_DEDUP_LOGS=0 UNIRL_RUN_ID="$run_name" \
  P0_AUDIT_ENABLE=1 P0_AUDIT_STRICT=1 P0_RECEIPT_DIR="$runtime_receipts" \
  P0_CAMPAIGN_ID="$campaign" P0_ARM_ID="$run_name" P0_PHASE=train \
  P0_EVIDENCE_CLASS=p2-prefetch-abba P0_LAUNCH_UUID="$run_name" \
  P0_RUN_NAME="$run_name" P0_NODE="$(hostname)" P0_SP_SIZE=2 \
  P0_EXPECTED_NUM_UPDATES=2 P0_SOURCE_COMMIT="$integration_commit" \
  P0_EXPECTED_SOURCE_COMMIT="$integration_commit" P0_DATASET_SHA256="$prompt_sha" \
  P0_CACHE_ROOT="$remote_root/no-persistent-cache" P0_REQUIRE_ZERO_CACHE_MISS=0 \
  P2_BASE_OVERLAY_SHA256="$base_overlay_sha" \
  P2_EXPECTED_PREFETCH="$([[ "$prefetch" == true ]] && echo 1 || echo 0)" \
  P2_RUNTIME_AGGREGATE="PYTHONPATH=$remote_root/runtime-overlay:$repo:$base_site $python_env/bin/python $remote_root/runtime-overlay/aggregate_runtime_events.py --event-root $runtime_receipts --p0-contract $p0_contract_path --campaign-id $campaign --run-name $run_name --treatment $treatment --output-dir $remote_root" \
  timeout --signal=INT --kill-after=900s "${timeout_s}s" bash -c '
    set -euo pipefail
    bash examples/run_experiment_single_node.sh \
      diffusion/minimax_h3/minimax_h3_t2va_trainside \
      num_devices=8 batch_size=8 num_rollouts=1 \
      "data_source.args.run.data_path=$1" data_source.args.run.seed=42 \
      +data_source.args.run.shuffle=false sampling.seed=42 sampling.height=768 \
      sampling.width=768 sampling.num_frames=124 sampling.num_inference_steps=10 \
      "sampling.sde_indices=[0,3,6]" sampling.eta=0.7 sampling.samples_per_prompt=4 \
      stack.num_updates_per_batch=2 stack.micro_batch_size=1 \
      ++stack.micro_planner._target_=unirl.train.stack.GroupInterleavedCountPlanner \
      ++backend.fsdp_cfg.sp_size=2 bundle.config.prompt_embedding_share_across_sp=true \
      bundle.config.prompt_embedding_cache_dir=null \
      bundle.config.prompt_embedding_cache_read_only=false \
      ++bundle.config.text_encoder_onload_for_embed=false \
      bundle.config.prompt_embedding_prefetch="$2" \
      bundle.config.prompt_embedding_prefetch_capacity=8 \
      +offload_train_during_reward=true "++load_dir=$3" ++save_interval=0 \
      logging.report_to_wandb=true logging.run_name="$4" logging.log_media=false \
      hydra.verbose=unirl.models.minimax_h3.text_embed hydra.job.chdir=false \
      "hydra.run.dir=$5"
    eval "$P2_RUNTIME_AGGREGATE"
  ' _ "$prompt_manifest" "$prefetch" "$load_dir" "$run_name" "$hydra_dir" \
  >"$log" 2>&1 </dev/null &
pid=$!
starttime=""
cmdline=""
for _ in $(seq 1 20); do
  if [[ -r /proc/$pid/stat ]]; then
    starttime=$(awk '{print $22}' "/proc/$pid/stat")
    cmdline=$(tr '\0' ' ' <"/proc/$pid/cmdline")
    break
  fi
  [[ -f "$status" ]] && break
  sleep 0.05
done
printf 'pid=%s\nstarttime=%s\ntoken=%s\ncampaign=%s\nrun=%s\nperiod=%s\ntreatment=%s\nprefetch=%s\ncontract_sha256=%s\np0_contract_sha256=%s\nprompt_manifest_sha256=%s\nfrozen_checkpoint_audit_sha256=%s\nfrozen_checkpoint_tree_sha256=%s\nfrozen_checkpoint_model_summary_sha256=%s\nfrozen_checkpoint_adapter_key_signature_sha256=%s\nplanner_contract_sha256=%s\ntwo_update_contract_sha256=%s\ncmdline=%s\n' \
  "$pid" "$starttime" "p2-prefetch-${campaign}-${run_name}" "$campaign" "$run_name" \
  "$period" "$treatment" "$prefetch" "$contract_sha" "$p0_sha" "$prompt_sha" \
  "$checkpoint_audit_sha" "$checkpoint_tree_sha" "$checkpoint_model_sha" \
  "$checkpoint_adapter_sha" "$planner_contract_sha" "$two_update_contract_sha" \
  "$cmdline" >"$identity"
printf 'LAUNCHED_PID=%s RUN=%s PERIOD=%s TREATMENT=%s LOG=%s STATUS=%s\n' \
  "$pid" "$run_name" "$period" "$treatment" "$log" "$status"
