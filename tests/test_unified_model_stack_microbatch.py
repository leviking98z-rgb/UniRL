import pytest

torch = pytest.importorskip("torch")

from unirl.algorithms import AlgorithmStepResult  # noqa: E402
from unirl.train.unified_model_stack import UnifiedModelTrainStack  # noqa: E402


class _Track:
    def __init__(self, values: torch.Tensor) -> None:
        self.conditions = {"values": values}
        self.segment = None
        self.advantages = torch.ones(values.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self.advantages.shape[0])

    def slice(self, start: int, end: int) -> "_Track":
        return _Track(self.conditions["values"][start:end])


class _MeanAlgorithm:
    def __init__(self) -> None:
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def compute_loss_and_backward(
        self,
        *,
        conditions,
        segment,
        advantages,
        training_progress,
        loss_scale,
    ) -> AlgorithmStepResult:
        del segment, advantages, training_progress
        loss = conditions["values"].float().mean() * self.scale
        (loss * loss_scale).backward()
        value = float(loss.detach().item())
        return AlgorithmStepResult(
            loss=value,
            metrics={"policy_loss": value},
            num_steps_or_tokens=len(conditions["values"]),
            has_backward=True,
        )


def _stack(*, ar_micro_batch_size=None, image_micro_batch_size=None):
    ar = _MeanAlgorithm()
    image = _MeanAlgorithm()
    stack = UnifiedModelTrainStack(
        fsdp_backend=object(),
        ar_algorithm=ar,
        image_algorithm=image,
        micro_batch_size=1,
        ar_micro_batch_size=ar_micro_batch_size,
        image_micro_batch_size=image_micro_batch_size,
        max_grad_norm=1.0,
    )
    return stack, ar, image


def test_ragged_micro_batches_preserve_sample_mean_gradient_and_loss() -> None:
    stack, algorithm, _ = _stack()
    track = _Track(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    slices = [(0, 2), (2, 4), (4, 5)]

    result, has_backward = stack._backward_track(
        "ar",
        track,
        slices,
        training_progress=0.0,
    )

    assert has_backward is True
    assert len(result.micros) == 3
    assert result.loss == pytest.approx(3.0)
    assert result.metrics["policy_loss"] == pytest.approx(3.0)
    assert float(algorithm.scale.grad) == pytest.approx(3.0)


def test_per_track_micro_batch_sizes_override_shared_fallback() -> None:
    stack, _, _ = _stack(ar_micro_batch_size=1, image_micro_batch_size=2)

    ar_steps = stack._optimizer_step_slices(4, micro_batch_size=stack.micro_batch_sizes["ar"])
    image_steps = stack._optimizer_step_slices(4, micro_batch_size=stack.micro_batch_sizes["image"])

    assert ar_steps == [[(0, 1), (1, 2), (2, 3), (3, 4)]]
    assert image_steps == [[(0, 2), (2, 4)]]
