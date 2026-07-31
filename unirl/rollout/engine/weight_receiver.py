"""Reusable engine-side delegation for common weight receiver capabilities."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class DelegatingTensorNCCLLoraReceiver:
    """Forward tensor, NCCL, and LoRA receiver calls to ``self._weight_sync``.

    SGLang's AR server intentionally ignores diffusion-oriented
    ``target_modules`` filters; set ``FORWARD_TARGET_MODULES = False`` there.
    """

    FORWARD_TARGET_MODULES = True
    _weight_sync: Any
    _weight_version: int

    def _receiver_target_modules(self, target_modules: Optional[List[str]]) -> dict:
        return {"target_modules": target_modules} if self.FORWARD_TARGET_MODULES else {}

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
        track_prefix: str = "",
    ) -> None:
        del track_prefix
        self._weight_sync.update_weights_from_tensor(
            serialized_named_tensors=serialized_named_tensors,
            load_format=load_format,
            flush_cache=flush_cache,
            **self._receiver_target_modules(target_modules),
        )
        self._weight_version += 1

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        track_prefix: str = "",
    ) -> None:
        del track_prefix
        self._weight_sync.init_weights_update_group(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )

    def update_weights_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        target_modules: Optional[List[str]] = None,
        flush_cache: bool = True,
        track_prefix: str = "",
    ) -> None:
        del track_prefix
        self._weight_sync.update_weights_from_distributed(
            names=names,
            dtypes=dtypes,
            shapes=shapes,
            group_name=group_name,
            flush_cache=flush_cache,
            **self._receiver_target_modules(target_modules),
        )
        self._weight_version += 1

    def destroy_weights_update_group(
        self,
        *,
        group_name: str,
        track_prefix: str = "",
    ) -> None:
        del track_prefix
        self._weight_sync.destroy_weights_update_group(group_name=group_name)

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, Any],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        self._weight_sync.set_lora_from_tensors(adapter_name, lora_tensors, peft_config=peft_config)

    @property
    def lora_dirty(self) -> bool:
        """Whether the adapter must be pushed before generation."""
        return self._weight_sync.lora_dirty


__all__ = ["DelegatingTensorNCCLLoraReceiver"]
