from __future__ import annotations

import sys
import types

import torch


def _install_vllm_omni_import_stubs() -> None:
    """Keep this worker-mixin unit test independent of a vLLM installation."""
    if "unirl.rollout.engine.vllm_omni.patches.runtime" not in sys.modules:
        runtime = types.ModuleType("unirl.rollout.engine.vllm_omni.patches.runtime")
        runtime.OmniTensorLoRARequest = object
        runtime.VLLMOmniHijack = type(
            "VLLMOmniHijack",
            (),
            {"hijack": staticmethod(lambda: None)},
        )
        sys.modules[runtime.__name__] = runtime


def test_loaded_lora_checksums_unwraps_ar_worker_manager(monkeypatch):
    _install_vllm_omni_import_stubs()
    from unirl.rollout.engine.vllm_omni.worker.ipc_receive_mixin import (
        BucketedIPCReceiveMixin,
    )

    monkeypatch.setattr(
        "unirl.distributed.weight_sync.transfer.checksum.fingerprint_tensor",
        lambda tensor: f"hash-{float(tensor.sum())}",
    )

    layer = types.SimpleNamespace(
        lora_a=torch.tensor([1.0, 2.0]),
        lora_b=torch.tensor([3.0]),
        bias=None,
        embeddings_tensor=None,
    )
    model = types.SimpleNamespace(loras={"q_proj": layer})
    adapter_manager = types.SimpleNamespace(_registered_adapters={1: model})
    worker_manager = types.SimpleNamespace(_adapter_manager=adapter_manager)
    worker = object.__new__(BucketedIPCReceiveMixin)
    worker.model_runner = types.SimpleNamespace(lora_manager=worker_manager)

    assert worker._diffrl_loaded_lora_checksums(1) == {
        "q_proj": {"lora_a": "hash-3.0", "lora_b": "hash-3.0"}
    }


def test_loaded_lora_checksums_keeps_direct_diffusion_manager(monkeypatch):
    _install_vllm_omni_import_stubs()
    from unirl.rollout.engine.vllm_omni.worker.ipc_receive_mixin import (
        BucketedIPCReceiveMixin,
    )

    monkeypatch.setattr(
        "unirl.distributed.weight_sync.transfer.checksum.fingerprint_tensor",
        lambda tensor: f"hash-{float(tensor.sum())}",
    )

    layer = types.SimpleNamespace(
        lora_a=torch.tensor([4.0]),
        lora_b=torch.tensor([5.0]),
        bias=None,
        embeddings_tensor=None,
    )
    model = types.SimpleNamespace(loras={"proj": layer})
    worker = object.__new__(BucketedIPCReceiveMixin)
    worker.lora_manager = types.SimpleNamespace(_registered_adapters={1: model})

    assert worker._diffrl_loaded_lora_checksums(1) == {
        "proj": {"lora_a": "hash-4.0", "lora_b": "hash-5.0"}
    }
