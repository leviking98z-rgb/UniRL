"""Compatibility facade for :mod:`unirl.distributed.peft`."""

from unirl.distributed.peft import (
    adapt_lora_for_sglang,
    adapt_lora_for_vllm,
    extract_lora_tensors,
    lora_targets_ep_experts,
    merged_state_dict,
    raw_state_dict,
)

__all__ = [
    "adapt_lora_for_sglang",
    "adapt_lora_for_vllm",
    "extract_lora_tensors",
    "lora_targets_ep_experts",
    "merged_state_dict",
    "raw_state_dict",
]
