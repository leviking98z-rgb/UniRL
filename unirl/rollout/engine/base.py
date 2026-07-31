"""Rollout engine base class for the ``Sample`` → ``Sample`` path.

Concrete engines take all runtime deps as ``__init__`` kwargs and complete
construction in one shot — no separate ``initialize(device)`` step. After
``__init__`` returns the engine is fully usable: model loaded, worker
subprocesses spawned, dist groups brought up. This matches the actor flow where
``_setup_distributed_env`` runs before the engine is built.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

from unirl.distributed.group.remote import Remote
from unirl.rollout.engine.lifecycle import DefaultRolloutMemoryLifecycle
from unirl.types.sample import Sample

#: The rollout batch contract (LIN-522). Single-turn engines return one ``Sample``;
#: the agentic engine returns a list of variable-depth trajectory ``Sample``s (one
#: per rollout). The broad generation method uses this union; the single-turn
#: subclass refines its return to ``Sample``.
RolloutOutput = Union[Sample, List[Sample]]


class BaseEngineConfig(ABC):
    """Marker base for all rollout engine config dataclasses.

    Used as the type annotation / base class for the engine config dataclasses.
    Each concrete engine config maps itself to its runtime engine class via
    :meth:`make_engine`.
    """

    def make_engine(self, **deps: Any) -> "BaseRolloutEngine":
        """Construct the runtime engine declared by this config.

        ``deps`` carry the runtime injections (``device``, ``strategy``,
        ``rank``, ``model_config``); the engine ctor contract is uniformly
        ``Engine(config=self, **deps)``. Subclasses override to import (lazily,
        so config modules stay importable without the engine's heavy optional
        deps) and return their engine class.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement make_engine()")


class BaseRolloutEngine(Remote, DefaultRolloutMemoryLifecycle, ABC):
    """Generation/control ABC composed with the default memory lifecycle."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def shutdown(self) -> None:
        """Release worker subprocesses and any other engine-owned resources."""

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(self, sample: Sample) -> RolloutOutput:
        """Synchronously run rollout generation for one request batch.

        Dispatch belongs to each concrete contract: single-turn engines use
        ``DP_SCATTER`` and return one ``Sample``; the agentic coordinator uses
        ``BROADCAST + RANK_ZERO`` and returns a trajectory list.
        """

    # ------------------------------------------------------------------
    # Control plane — sync methods reached via the raw ``Worker.call`` RPC (the
    # un-decorated weight-sync pattern), so they interleave with an in-flight
    # ``generate`` on a threaded Worker (``worker_max_concurrency>1``).
    # ------------------------------------------------------------------

    def abort(self, ids: Optional[List[str]] = None) -> List[Sample]:
        """Best-effort cancel of in-flight generation; return any partials.

        Default no-op (``[]``). Engines whose backend supports it (SGLang) cancel
        running requests; sync/batch backends can only drop not-yet-started work.
        """
        del ids
        return []

    def pause(self) -> None:
        """Stop admitting new generation (best-effort). Default no-op."""

    def resume(self) -> None:
        """Resume generation after :meth:`pause`. Default no-op."""


class BaseSingleTurnRolloutEngine(BaseRolloutEngine, ABC):
    """Nominal contract for engines that fill and return one ``Sample``.

    The class intentionally does not prescribe batching or provide a batching
    wrapper. Each engine owns those semantics and applies its own
    ``@distributed`` decorator. The contract is synchronous. An engine meant to
    serve as an agentic inner must make ``generate`` safe for CONCURRENT
    callers (the agentic drain calls it from one thread per trajectory and
    relies on the backend batching the in-flight requests together); an engine
    that cannot serve concurrently serializes internally instead.
    """

    #: Policy weight version the current weights correspond to (bumped on each
    #: weight sync; stamped onto generated Parts by :meth:`_stamp_weight_version`).
    _weight_version: int = 0

    @abstractmethod
    def generate(self, sample: Sample) -> Sample:
        """Synchronously fill and return one request ``Sample``."""

    def _stamp_weight_version(self, sample: Sample) -> Sample:
        """Stamp ``self._weight_version`` onto the frontier (last) gen Part."""
        v = getattr(self, "_weight_version", None)
        if v is None or not sample.parts:
            return sample
        gen = sample.parts[-1].fill(weight_version=int(v))
        return sample.with_parts([*sample.parts[:-1], gen])


__all__ = ["BaseRolloutEngine", "BaseSingleTurnRolloutEngine", "RolloutOutput"]
