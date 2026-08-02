from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from unirl.models.hunyuan_image3 import pipeline as pipeline_module  # noqa: E402
from unirl.models.hunyuan_image3.conditions import (  # noqa: E402
    HunyuanImage3DiffusionConditions,
    HunyuanImage3FusedMultimodalCondition,
)
from unirl.models.hunyuan_image3.config import HunyuanImage3PipelineConfig  # noqa: E402
from unirl.models.hunyuan_image3.diffusion import HunyuanImage3DiffusionStage  # noqa: E402
from unirl.sde.kernels import FlowSDEStrategy  # noqa: E402
from unirl.types.conditions import ImageEmbedCondition, ImageLatentCondition  # noqa: E402
from unirl.types.segments.latent import LatentSegment  # noqa: E402


class _FakeBundle(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.7, dtype=torch.float32))


class _FakeStep:
    """Minimal differentiable stand-in for the HI3 transformer + SDE step."""

    def __init__(self) -> None:
        self.calls = 0

    def step_with_logp(
        self,
        model,
        conditions,
        *,
        strategy,
        sample,
        prev_sample,
        sigma,
        sigma_next,
        guidance_scale,
        eta,
        sigma_max,
        step_index,
    ):
        del strategy, sigma_next, guidance_scale, eta, sigma_max, step_index
        self.calls += 1
        batch = int(sample.shape[0])
        sigma = sigma.to(dtype=sample.dtype)
        if sigma.dim() == 0:
            sigma = sigma.expand(batch)
        cond = conditions.fused.input_ids[:, 0].to(dtype=sample.dtype)
        tail = [1] * (sample.dim() - 1)
        mean = sample * model.scale + cond.view(batch, *tail) * 0.03 + sigma.view(batch, *tail) * 0.07
        log_prob = -((prev_sample - mean) ** 2).flatten(1).mean(dim=1)
        return prev_sample, log_prob, mean


def _conditions() -> HunyuanImage3DiffusionConditions:
    input_ids = torch.tensor([[2, 3, 4], [7, 8, 9]], dtype=torch.long)
    batch, length = input_ids.shape
    fused = HunyuanImage3FusedMultimodalCondition(
        input_ids=input_ids,
        attention_mask=torch.ones(batch, 1, length, length, dtype=torch.bool),
        position_ids=torch.arange(length).expand(batch, -1),
        rope_cache=(
            torch.arange(batch * length * 2, dtype=torch.float32).view(batch, length, 2),
            torch.arange(batch * length * 2, dtype=torch.float32).view(batch, length, 2) + 1,
        ),
        gen_image_mask=torch.tensor([[False, True, True], [False, True, True]]),
        gen_timestep_scatter_index=torch.zeros(batch, 1, dtype=torch.long),
        cond_vae_image_mask=torch.tensor([[True, False, False], [True, False, False]]),
        cond_vit_image_mask=torch.tensor([[True, False, False], [True, False, False]]),
        cond_timestep_scatter_index=torch.zeros(batch, 1, dtype=torch.long),
        prompt_lengths=torch.tensor([2, 3], dtype=torch.long),
    )
    return HunyuanImage3DiffusionConditions(
        fused=fused,
        cond_vae=ImageLatentCondition(latents=torch.arange(batch * 2, dtype=torch.float32).view(batch, 2)),
        cond_vit=ImageEmbedCondition(
            embeds=torch.arange(batch * 6, dtype=torch.float32).view(batch, 3, 2),
            attn_mask=torch.ones(batch, 3, dtype=torch.bool),
            spatial_shapes=[(1, 3), (3, 1)],
        ),
        cond_timestep=torch.tensor([0.1, 0.2], dtype=torch.float32),
        tokenizer_output={"opaque": "preserved"},
    )


def _segment() -> LatentSegment:
    # Store positions 0..3 so transitions 0->1 and 2->3 are replayable.
    return LatentSegment(
        latents=torch.arange(2 * 4 * 1 * 2 * 2, dtype=torch.float32).view(2, 4, 1, 2, 2) / 10,
        sigmas=torch.tensor([0.9, 0.6, 0.3, 0.1], dtype=torch.float32),
        indices=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        sde_indices=torch.tensor([0, 2], dtype=torch.long),
        sde_logp=torch.zeros(2, 2, dtype=torch.float32),
    )


def _stage(*, batch_replay_steps: bool):
    model = _FakeBundle()
    step = _FakeStep()
    stage = HunyuanImage3DiffusionStage(
        model=model,
        step=step,
        strategy=FlowSDEStrategy(),
        autocast_precision="fp32",
        trajectory_precision="fp32",
        logprob_precision="fp32",
        batch_replay_steps=batch_replay_steps,
    )
    return stage, model, step


def test_hi3_batch_replay_defaults_off() -> None:
    cfg = HunyuanImage3PipelineConfig(pretrained_model_ckpt_path="unused")
    assert cfg.batch_replay_steps is False


def test_hi3_pipeline_threads_batch_replay_config_to_stage(monkeypatch) -> None:
    for name in (
        "HunyuanImage3TextEmbedStage",
        "HunyuanImage3VAEDecodeStage",
        "HunyuanImage3VAEEncodeStage",
        "HunyuanImage3ARStage",
        "HunyuanImage3VitEncodeStage",
    ):
        monkeypatch.setattr(pipeline_module, name, lambda *args, **kwargs: object())

    cfg = HunyuanImage3PipelineConfig(
        pretrained_model_ckpt_path="unused",
        batch_replay_steps=True,
    )
    pipeline = pipeline_module.HunyuanImage3Pipeline._assemble(
        _FakeBundle(),
        config=cfg,
        strategy=FlowSDEStrategy(),
    )
    assert pipeline.diffusion.batch_replay_steps is True


def test_hi3_batch_replay_matches_serial_values_order_and_gradients() -> None:
    conditions = _conditions()
    segment = _segment()
    params = SimpleNamespace(guidance_scale=1.0, eta=1.0)
    # Deliberately reverse the stored SDE order: the public result must follow
    # requested target order, not silently sort it.
    target = [2, 0]

    slow, slow_model, slow_step = _stage(batch_replay_steps=False)
    fast, fast_model, fast_step = _stage(batch_replay_steps=True)
    slow_result = slow.replay(conditions, segment=segment, params=params, step_indices=target)
    fast_result = fast.replay(conditions, segment=segment, params=params, step_indices=target)

    assert slow_step.calls == len(target)
    assert fast_step.calls == 1
    assert slow_result.log_probs.shape == fast_result.log_probs.shape == (2, 2)
    assert slow_result.prev_sample_means.shape == fast_result.prev_sample_means.shape == (2, 2, 1, 2, 2)
    torch.testing.assert_close(fast_result.log_probs, slow_result.log_probs)
    torch.testing.assert_close(fast_result.prev_sample_means, slow_result.prev_sample_means)

    slow_loss = slow_result.log_probs.sum() + slow_result.prev_sample_means.square().mean()
    fast_loss = fast_result.log_probs.sum() + fast_result.prev_sample_means.square().mean()
    slow_loss.backward()
    fast_loss.backward()
    torch.testing.assert_close(fast_loss, slow_loss)
    torch.testing.assert_close(fast_model.scale.grad, slow_model.scale.grad)


def test_hi3_batch_replay_tiles_every_batched_condition_in_step_major_order() -> None:
    stage, _, _ = _stage(batch_replay_steps=True)
    conditions = _conditions()
    tiled = stage._tile_conditions(conditions, 2, sample_batch_size=2)

    assert torch.equal(tiled.fused.input_ids, torch.cat([conditions.fused.input_ids] * 2))
    assert torch.equal(tiled.fused.rope_cache[0], torch.cat([conditions.fused.rope_cache[0]] * 2))
    assert torch.equal(tiled.cond_vae.latents, torch.cat([conditions.cond_vae.latents] * 2))
    assert torch.equal(tiled.cond_vit.embeds, torch.cat([conditions.cond_vit.embeds] * 2))
    assert tiled.cond_vit.spatial_shapes == conditions.cond_vit.spatial_shapes * 2
    assert torch.equal(tiled.cond_timestep, torch.cat([conditions.cond_timestep] * 2))
    assert tiled.tokenizer_output is conditions.tokenizer_output
