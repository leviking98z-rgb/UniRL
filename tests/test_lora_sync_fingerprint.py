from unirl.distributed.weight_sync.lora.base import LoraWeightSyncBase


def test_lora_checksum_fingerprint_is_stable_and_version_sensitive():
    first = LoraWeightSyncBase._checksum_fingerprint(["a", "b"], ["c"])
    same = LoraWeightSyncBase._checksum_fingerprint(["a", "b"], ["c"])
    changed = LoraWeightSyncBase._checksum_fingerprint(["a", "b"], ["d"])

    assert first == same
    assert first != changed
    assert len(first) == 16
