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

        Returns ``(routing_npy, forward_ntok)``: routing is a NumPy int16 array
        ``[n_layers, total_tok, top_k]`` (or None). We return NumPy (not a torch
        tensor) because collective_rpc's serialization turns torch tensors into
        nested Python lists (losing the tensor type on the driver side); a NumPy
        array round-trips as an array. int16 halves the wire size (expert ids <
        64k). forward_ntok is the per-forward token-count list.
        """
        try:
            from unirl.rollout.engine.vllm_omni.patches.moe_route_capture import drain_global

            routing, ntok = drain_global()
            if routing is None:
                return None, []
            # torch int64 [L, T, K] -> numpy int16 (expert ids are small)
            return routing.to("cpu").numpy().astype("int16"), ntok
        except Exception:
            return None, []


__all__ = ["HI3ARWeightSyncExtension"]
