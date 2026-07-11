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

        Returns ``(shape_tuple, raw_bytes, forward_ntok)`` or ``(None, None, [])``.

        WHY bytes (Bug3c): vllm's collective_rpc serializes the return value with
        ``msgspec.msgpack``. msgpack has NO numpy/torch support, so a returned
        tensor/ndarray is coerced into nested Python lists (losing the type on the
        driver side — that's the bug we hit). msgpack DOES pass ``bytes``
        through verbatim, so we ship the routing as int16 raw bytes + its shape,
        and the driver rebuilds the tensor with np.frombuffer. No collective_rpc
        tensor-serialization, no NCCL group, no shared file.
        """
        try:
            from unirl.rollout.engine.vllm_omni.patches.moe_route_capture import drain_global

            routing, ntok = drain_global()
            if routing is None:
                return None, None, []
            arr = routing.to("cpu").numpy().astype("int16")  # expert ids < 64k
            return tuple(int(s) for s in arr.shape), arr.tobytes(), ntok
        except Exception:
            return None, None, []


__all__ = ["HI3ARWeightSyncExtension"]
