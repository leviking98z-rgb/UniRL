"""Pure helpers for slime-style partial rollout (interrupt-at-sync + carry).

Stateless functions over RolloutTrack/RolloutReq so they unit-test without a
GPU. A rollout sequence that is interrupted (finish_reason 'abort') or hits the
per-round length cap ('length') is *carried*: its prompt + tokens-so-far are
re-submitted next weight version (continuation), until it finishes naturally
('stop'/'eos'). Whole GENERATIONS are carried as a unit, so GRPO groups never
split. The stateful orchestration (abort POST, in-flight carry, relaunch) lives
in AsyncARTrainer.
"""
from __future__ import annotations

from typing import List

import torch

from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutTrack
from unirl.types.segments.text import TextSegment
from unirl.types.primitives import Texts
from unirl.distributed.tensor.ref import hydrate

# A sequence is DONE only when the model chose to stop. 'length' (hit the
# per-round new-token cap) and 'abort' (interrupted for a weight sync) both mean
# 'continue next version'.
# Only an INTERRUPTED sequence is carried. "length" (hit max_new_tokens) is
# truncated-done; "stop"/"eos" finished naturally. Only "abort" continues.
CARRY_REASONS = frozenset({"abort"})


def _per_sample(track: RolloutTrack):
    seg = track.segment
    cu = seg.cu_seqlens
    n = len(track.sample_ids)
    tok = hydrate(seg.tokens)
    lp = hydrate(seg.log_probs)
    toks = [tok[cu[i] : cu[i + 1]].clone() for i in range(n)]
    lps = [lp[cu[i] : cu[i + 1]].clone() for i in range(n)]
    return toks, lps


def _decoded_list(track: RolloutTrack, n: int) -> list:
    d = getattr(track, "decoded", None)
    if d is not None and getattr(d, "texts", None) is not None:
        return list(d.texts)
    return [""] * n


def concretize(track: RolloutTrack) -> RolloutTrack:
    """Rebuild a track with its segment hydrated to concrete tensors, so a carried
    partial survives the weight sync (lazy TensorRefs may be invalidated by it)."""
    toks, lps = _per_sample(track)
    return RolloutTrack(
        sample_ids=list(track.sample_ids),
        parent_ids=list(track.parent_ids) if track.parent_ids else None,
        finish_reasons=list(track.finish_reasons) if track.finish_reasons else None,
        conditions=dict(track.conditions) if track.conditions else {},
        decoded=Texts(texts=_decoded_list(track, len(track.sample_ids))),
        segment=TextSegment.pack(tokens=toks, log_probs=lps),
    )


def unfinished_indices(track: RolloutTrack) -> List[int]:
    fr = track.finish_reasons or []
    return [i for i, f in enumerate(fr) if str(f) in CARRY_REASONS]


def is_complete(track: RolloutTrack) -> bool:
    return len(unfinished_indices(track)) == 0


def build_continuation_req(req: RolloutReq, track: RolloutTrack, idx: List[int], dp_size: int = 1) -> RolloutReq:
    """Subset  to the unfinished samples , attaching their tokens-so-far
    as continuation_token_ids (input_ids = prompt + these next round).

    The sample count is padded up to a multiple of  by duplicating the
    last unfinished sample, so the continuation still splits evenly over the
    rollout DP shards (DP_SCATTER divisibility). The duplicate rows are ignored
    on merge (merge_continuation only consumes the first len(idx) results)."""
    toks, _ = _per_sample(track)
    pad = (-len(idx)) % max(1, int(dp_size)) if idx else 0
    idx_padded = list(idx) + [idx[-1]] * pad if idx else list(idx)
    sub = req.select(torch.tensor(idx_padded, dtype=torch.long))
    sub.continuation_token_ids = [list(toks[i].tolist()) for i in idx_padded]
    return sub


def merge_continuation(carried: RolloutTrack, cont: RolloutTrack, idx: List[int]) -> RolloutTrack:
    """Fold the continuation result  (new tokens for the  subset, in
    that order) back into the full  track. Returns a new full track with
    per-sample tokens/logprobs concatenated and finish_reasons updated."""
    c_toks, c_lps = _per_sample(carried)
    n_toks, n_lps = _per_sample(cont)
    fr = list(carried.finish_reasons or [])
    cont_fr = list(cont.finish_reasons or [])
    c_dec = _decoded_list(carried, len(carried.sample_ids))
    cont_dec = _decoded_list(cont, len(idx))
    for j, i in enumerate(idx):
        c_toks[i] = torch.cat([c_toks[i], n_toks[j]])
        c_lps[i] = torch.cat([c_lps[i], n_lps[j]])
        fr[i] = str(cont_fr[j]) if j < len(cont_fr) else "unknown"
        c_dec[i] = c_dec[i] + (cont_dec[j] if j < len(cont_dec) else "")
    return RolloutTrack(
        sample_ids=list(carried.sample_ids),
        parent_ids=list(carried.parent_ids) if carried.parent_ids else None,
        finish_reasons=fr,
        conditions=dict(carried.conditions) if carried.conditions else {},
        decoded=Texts(texts=c_dec),
        segment=TextSegment.pack(tokens=c_toks, log_probs=c_lps),
    )
