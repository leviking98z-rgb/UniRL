from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from unirl.distributed.group.dispatch import (  # noqa: E402
    DISTRIBUTED_CONFIG_ATTR,
    Dispatch,
    _dispatch_dp_scatter,
    _dispatch_dp_scatter_independent,
    resolve_backward_dispatch_mode,
)
from unirl.train.unified_model_stack import UnifiedModelTrainStack  # noqa: E402
from unirl.types.rollout_resp import RolloutTrack  # noqa: E402


def _fake_dp_handle(dp_size: int, *, replicas_per_dp_rank: int = 1):
    return SimpleNamespace(
        dp_size=dp_size,
        world_size=dp_size * replicas_per_dp_rank,
        rank_infos=[
            SimpleNamespace(dp_rank=dp_rank) for dp_rank in range(dp_size) for _ in range(replicas_per_dp_rank)
        ],
    )


def test_independent_dp_scatter_shards_each_top_level_batch() -> None:
    ar = torch.arange(16)
    image = torch.arange(32)

    shards = _dispatch_dp_scatter_independent(
        _fake_dp_handle(8),
        (ar, image),
        {"training_progress": 0.5},
        batch_size=16,
    )

    assert len(shards) == 8
    for rank, (shard_args, shard_kwargs) in enumerate(shards):
        torch.testing.assert_close(shard_args[0], ar[rank * 2 : (rank + 1) * 2])
        torch.testing.assert_close(shard_args[1], image[rank * 4 : (rank + 1) * 4])
        assert shard_kwargs == {"training_progress": 0.5}


def test_unwrapped_mismatched_batch_keeps_existing_broadcast_semantics() -> None:
    ar = torch.arange(16)
    image = torch.arange(32)

    shards = _dispatch_dp_scatter(_fake_dp_handle(8), (ar, image), {}, batch_size=16)

    assert all(shard_args[1] is image for shard_args, _ in shards)


def test_independent_dp_scatter_validates_every_top_level_batch() -> None:
    with pytest.raises(
        ValueError,
        match=r"arg\[1\] batch_size=10 not divisible by dp_size=8 under DP_SCATTER_INDEPENDENT",
    ):
        _dispatch_dp_scatter_independent(
            _fake_dp_handle(8),
            (torch.arange(16), torch.arange(10)),
            {},
            batch_size=16,
        )


def test_independent_dp_scatter_chunks_tracks_once_and_replicates_within_dp_rank() -> None:
    ar = RolloutTrack(
        sample_ids=[f"ar-{index}" for index in range(4)],
        advantages=torch.arange(4, dtype=torch.float32),
    )
    image = RolloutTrack(
        sample_ids=[f"image-{index}" for index in range(8)],
        advantages=torch.arange(8, dtype=torch.float32),
    )

    shards = _dispatch_dp_scatter_independent(
        _fake_dp_handle(2, replicas_per_dp_rank=2),
        (ar, image),
        {},
        batch_size=4,
    )

    assert len(shards) == 4
    for worker_pair in ((0, 1), (2, 3)):
        first, second = (shards[index][0] for index in worker_pair)
        assert first[0].sample_ids == second[0].sample_ids
        assert first[1].sample_ids == second[1].sample_ids
    assert [sample for index in (0, 2) for sample in shards[index][0][0].sample_ids] == ar.sample_ids
    assert [sample for index in (0, 2) for sample in shards[index][0][1].sample_ids] == image.sample_ids


def test_independent_dp_scatter_rejects_auto_backward() -> None:
    with pytest.raises(ValueError, match=r"DP_SCATTER_INDEPENDENT.*does not support auto-backward"):
        resolve_backward_dispatch_mode(
            "train_track",
            Dispatch.DP_SCATTER_INDEPENDENT,
            [SimpleNamespace(pp_size=1)],
        )


def test_unified_train_track_opts_into_independent_dp_scatter() -> None:
    config = getattr(UnifiedModelTrainStack.train_track, DISTRIBUTED_CONFIG_ATTR)
    assert config["dispatch_mode"] is Dispatch.DP_SCATTER_INDEPENDENT
