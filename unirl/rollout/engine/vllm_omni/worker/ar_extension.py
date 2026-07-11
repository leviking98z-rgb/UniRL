"""Worker-extension class installed on the HI3 AR stage of vllm-omni.

Composes:

- ``BucketedIPCReceiveMixin`` — bucketed CUDA-IPC ``update_weights_from_ipc``
  + LoRA-bucket dispatch + ``VLLMOmniHijack`` install in ``__new__``.
- ``NcclBroadcastReceiveMixin`` — SGLang-shape NCCL primitives
  (``init_weights_update_group``, ``update_weights_from_distributed``,
  ``destroy_weights_update_group``).
- ``HI3ARWorkerExtension`` (``compat/tokenizer``) — preserves the
  module-import side effect that patches ``PreTrainedTokenizer.convert_tokens_to_ids``
  for the Base ckpt's missing ratio tokens.

The AR worker (``GPUARWorker`` → ``OmniGPUWorkerBase`` → upstream
``vllm.v1.worker.gpu_worker.Worker``) already inherits upstream's
``init_weight_transfer_engine`` / ``update_weights(update_info)`` for
the ``WeightTransferEngine`` path. We use the SGLang-shape NCCL methods
on top of (not instead of) those — both are reachable via collective_rpc.
"""

from __future__ import annotations

from unirl.rollout.engine.vllm_omni.patches.compat_tokenizer import HI3ARWorkerExtension
from unirl.rollout.engine.vllm_omni.worker.ipc_receive_mixin import (
    BucketedIPCReceiveMixin,
)
from unirl.rollout.engine.vllm_omni.worker.nccl_receive_mixin import (
    NcclBroadcastReceiveMixin,
)


class HI3ARWeightSyncExtension(
    BucketedIPCReceiveMixin,
    NcclBroadcastReceiveMixin,
    HI3ARWorkerExtension,
):
    """Receive-side extension for the HI3 AR stage."""

    def _diffrl_drain_routing(self):
        """Drain the process-global MoE route-capture buffer on this worker.

        Called by the driver via ``collective_rpc`` after each AR generate (when
        UNIRL_MOE_ROUTE_CAPTURE=1). Returns ``(routing, forward_ntok)``: routing
        is a CPU int64 tensor ``[n_layers, total_tok, top_k]`` (or None) and
        forward_ntok is the per-forward token-count list the driver uses to split
        routing back to per-request slices. Draining resets the buffer for the
        next rollout. Inert ((None, [])) if capture disabled or this process never
        ran the AR MoE.
        """
        try:
            from unirl.rollout.engine.vllm_omni.patches.moe_route_capture import drain_global

            return drain_global()
        except Exception:
            return None, []


__all__ = ["HI3ARWeightSyncExtension"]
