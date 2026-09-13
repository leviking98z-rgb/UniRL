"""MiniMax-H3 text embedding stage -- Qwen3-VL layer-50 hidden states."""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import tempfile
import threading
import time
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import timedelta
from functools import cache
from typing import TYPE_CHECKING, Iterator, List, Sequence

import torch

from unirl.config.require import require
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts
from unirl.utils.run_id import resolve_run_id

from .prefetch import BoundedPrefetcher, PrefetchCancelledError
from .vendor import MINIMAX_H3_TEXT_ENCODER_LAYER

if TYPE_CHECKING:
    from .bundle import MiniMaxH3Bundle

logger = logging.getLogger(__name__)
_EMBED_SYNC_TIMEOUT = timedelta(minutes=30)
_DISK_CACHE_SCHEMA = "unirl:minimax-h3:qwen3-vl:text-embedding:v1"
_PREFETCH_SCHEMA = "unirl:minimax-h3:qwen3-vl:text-embedding-prefetch:v1"
# One entry is [1, tokens, hidden] in the encoder's own dtype — a few MB for a
# typical prompt, so this bounds the CPU-resident cache at a few hundred MB.
_PROMPT_CACHE_SIZE = 64


@dataclass(frozen=True)
class _PrefetchedEmbedding:
    """One validated prompt identity and its CPU embedding."""

    digest: str
    token_ids: tuple[int, ...]
    embedding: torch.Tensor


@cache
def _residency_lock_path() -> str:
    """Return a per-run, per-user node-local conditioner lock path."""
    digest = hashlib.sha256(f"{os.getuid()}:{resolve_run_id()}".encode()).hexdigest()[:16]
    # Left behind on purpose: the file is empty and node-local, and unlinking it
    # would race the sibling ranks still holding the lock through it.
    return os.path.join(tempfile.gettempdir(), f"unirl_minimax_h3_onload_{digest}.lock")


@contextmanager
def _serialize_node_residency() -> Iterator[None]:
    """Admit one rank at a time to the conditioner residency window on this node."""
    if os.environ.get("UNIRL_MINIMAX_H3_ONLOAD_SERIALIZE", "1") == "0":
        yield
        return

    with open(_residency_lock_path(), "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@cache
def _conditioner_compute_lock_path(run_identity: str, cache_dir: str, checkpoint_identity: str) -> str:
    """Return the per-run node-local lock path for one persistent cache."""
    identity = (
        os.getuid(),
        run_identity,
        os.path.realpath(cache_dir),
        checkpoint_identity,
    )
    digest = hashlib.sha256(repr(identity).encode()).hexdigest()[:16]
    return os.path.join(tempfile.gettempdir(), f"unirl_minimax_h3_compute_{digest}.lock")


@contextmanager
def _serialize_node_conditioner_compute(cache_dir: str, checkpoint_identity: str) -> Iterator[None]:
    """Serialize persistent-cache cold fills by ranks sharing one launch."""
    if os.environ.get("UNIRL_MINIMAX_H3_ONLOAD_SERIALIZE", "1") == "0":
        yield
        return

    run_identity = (
        os.environ.get("UNIRL_RUN_ID")
        or os.environ.get("RAY_JOB_ID")
        or (
            f"torchrun:{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
            if "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ
            else f"pid:{os.getpid()}"
        )
    )
    lock_path = _conditioner_compute_lock_path(run_identity, cache_dir, checkpoint_identity)
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class MiniMaxH3TextEmbedStage:
    """Encode prompts into the conditioning MiniMax-H3 was trained on."""

    def __init__(self, bundle: "MiniMaxH3Bundle") -> None:
        self.text_encoder = bundle.text_encoder
        self.processor = bundle.processor
        self.tokenizer = bundle.tokenizer
        self.dtype = bundle.dtype
        self.device = bundle.device
        self.vae = bundle.vae
        self.audio_vae = bundle.audio_vae
        self._onload_for_embed = bundle.text_encoder_onload_for_embed
        cache_dir = getattr(bundle, "prompt_embedding_cache_dir", None)
        self._disk_cache_dir = (
            os.path.abspath(os.path.expanduser(os.fspath(cache_dir))) if cache_dir is not None else None
        )
        self._disk_cache_read_only = bool(getattr(bundle, "prompt_embedding_cache_read_only", False))
        self._share_across_sp = bool(getattr(bundle, "prompt_embedding_share_across_sp", False))
        prefetch_requested = bool(getattr(bundle, "prompt_embedding_prefetch", False))
        prefetch_capacity = int(getattr(bundle, "prompt_embedding_prefetch_capacity", 8))
        self._checkpoint_identity = str(getattr(bundle, "text_encoder_checkpoint_identity", bundle.pretrained_path))
        self._text_encoder_dtype = getattr(bundle, "text_encoder_dtype", None)
        if self._text_encoder_dtype is None:
            self._text_encoder_dtype = next(self.text_encoder.parameters()).dtype
        self._expected_hidden_size = self._resolve_hidden_size()
        # The encoder is frozen and prompt-only. Trainside forward_batch_size=1
        # invokes pipeline.generate once per sibling sample, so cache on CPU to
        # avoid repeating a 32B conditioner forward for the same prompt.
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._embedding_sync_group = None
        self._conditioner_lock = threading.Lock()
        self._prefetch = (
            BoundedPrefetcher[str, _PrefetchedEmbedding](
                prefetch_capacity,
                thread_name="minimax-h3-prompt-prefetch",
            )
            if prefetch_requested and self._prefetch_compatible
            else None
        )
        if prefetch_requested and self._prefetch is None:
            logger.info(
                "MiniMax-H3 prompt prefetch disabled: it requires a CPU-resident conditioner, "
                "text_encoder_onload_for_embed=False, and no persistent disk cache"
            )

    @property
    def _encoder_device(self) -> torch.device:
        return next(self.text_encoder.parameters()).device

    @property
    def _decoder(self):
        """The decoder stack that owns ``.layers``."""
        model = self.text_encoder.model
        return getattr(model, "language_model", model)

    @property
    def _prefetch_compatible(self) -> bool:
        """Return whether background conditioner execution is CPU-only and local."""
        return self._encoder_device.type == "cpu" and not self._onload_for_embed and self._disk_cache_dir is None

    @property
    def prefetch_enabled(self) -> bool:
        """Return whether next-microbatch prompt prefetch is active."""
        return self._prefetch is not None and not self._prefetch.closed

    @torch.no_grad()
    def prefetch(self, texts: Texts) -> bool:
        """Queue CPU embeddings without entering distributed collectives."""
        prefetcher = self._prefetch
        if prefetcher is None or prefetcher.closed:
            return False
        prompts = list(dict.fromkeys(texts.texts))
        if not prompts:
            return False
        if self._share_across_sp and not self._is_sp_source_rank():
            return True

        submitted = False
        for prompt in prompts:
            if prompt in self._cache:
                continue
            token_ids = self._tokenize(prompt)
            key = self._prefetch_key(token_ids)
            accepted = prefetcher.submit(
                key,
                lambda prompt=prompt, token_ids=token_ids, key=key: self._encode_prefetch(
                    prompt,
                    token_ids,
                    key,
                ),
            )
            submitted = submitted or accepted
        return submitted

    def shutdown(self) -> None:
        """Stop prompt prefetch after finishing an active CPU forward."""
        if self._prefetch is not None:
            self._prefetch.shutdown()

    @torch.no_grad()
    def embed(self, texts: Texts) -> TextEmbedCondition:
        """Encode one batch of prompts into a ``TextEmbedCondition``."""
        # ``Texts.texts`` is the raw list[str]; ``Texts.to_list()`` returns
        # list[Text] dataclass wrappers, which the tokenizer rejects. Same
        # accessor ltx2 and wan21 use.
        prompts: List[str] = list(texts.texts)
        require(len(prompts) > 0, "MiniMaxH3TextEmbedStage: no prompts to embed")

        num_layers = len(self._decoder.layers)
        require(
            num_layers > MINIMAX_H3_TEXT_ENCODER_LAYER,
            f"MiniMaxH3TextEmbedStage: MiniMax-H3 conditions on hidden_states[{MINIMAX_H3_TEXT_ENCODER_LAYER}] of "
            f"its Qwen3-VL conditioner, which needs more than {MINIMAX_H3_TEXT_ENCODER_LAYER} decoder layers, but "
            f"the loaded conditioner has {num_layers}.",
        )

        self._ensure_embedding_sync_group()
        unique_prompts = list(dict.fromkeys(prompts))
        started = time.perf_counter()
        error: BaseException | None = None
        condition: TextEmbedCondition | None = None
        memory_hits = disk_hits = misses = shared_hits = 0
        shared_source = False
        try:
            sp_group = self._resolve_sp_share_group()
            if sp_group is None:
                resolved, memory_hits, disk_hits, misses = self._resolve_local(unique_prompts)
            else:
                resolved, memory_hits, disk_hits, misses, shared_hits, shared_source = self._resolve_shared(
                    unique_prompts, sp_group
                )
            for prompt in unique_prompts:
                if prompt in self._cache:
                    self._cache.move_to_end(prompt)
            embeds = [resolved[prompt].to(device=self.device, dtype=self.dtype) for prompt in prompts]

            lengths = {int(e.shape[1]) for e in embeds}
            require(
                len(lengths) == 1,
                f"MiniMaxH3TextEmbedStage: prompts tokenized to differing lengths {sorted(lengths)}. The packed "
                "sequence geometry must be identical across the batch (LatentSegment stores latents in a CONCAT "
                "field), so a mixed-length batch cannot be packed. Pad or group prompts by token length upstream.",
            )
            text_embeds = torch.cat(embeds, dim=0)
            condition = TextEmbedCondition(
                embeds=text_embeds,
                attn_mask=torch.ones(text_embeds.shape[:2], dtype=torch.bool, device=text_embeds.device),
            )
        except BaseException as exc:
            error = exc

        if self._needs_embedding_status_sync():
            if not self._sync_embedding_status(error is None):
                if error is not None:
                    raise error
                raise RuntimeError("MiniMax-H3 prompt embedding failed on another rank")
        elif error is not None:
            raise error
        assert condition is not None
        self._synchronize_embedding_ranks()
        logger.debug(
            "MiniMaxH3 text embeds: prompts=%d cache_hits=%d misses=%d onload=%s elapsed_s=%.3f "
            "memory_hits=%d disk_hits=%d shared_hits=%d shared_source=%s",
            len(prompts),
            memory_hits + disk_hits + shared_hits,
            misses,
            self._onload_for_embed,
            time.perf_counter() - started,
            memory_hits,
            disk_hits,
            shared_hits,
            shared_source,
        )
        return condition

    def _resolve_local(self, prompts: Sequence[str]) -> tuple[dict[str, torch.Tensor], int, int, int]:
        """Resolve prompts from the rank-local caches or conditioner."""
        resolved = {prompt: self._cache[prompt] for prompt in prompts if prompt in self._cache}
        memory_hits = len(resolved)
        missing = [prompt for prompt in prompts if prompt not in resolved]
        prefetched = self._take_prefetched(missing, resolved)
        memory_hits += prefetched
        missing = [prompt for prompt in missing if prompt not in resolved]
        disk_hits = 0
        misses = len(missing)
        if missing and self._disk_cache_dir is not None:
            disk_hits, misses = self._resolve_with_disk_cache(missing, resolved)
        elif missing:
            self._encode_missing(missing, resolved)
        return resolved, memory_hits, disk_hits, misses

    def _resolve_sp_share_group(self):
        """Return the active Ulysses group when SP prompt sharing is enabled."""
        if not self._share_across_sp:
            return None

        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return None

        group = None
        sp_size = 1
        state_error: BaseException | None = None
        try:
            from unirl.train.backend.veomni import _compat

            _compat.ensure_installed()
            from veomni.distributed.parallel_state import get_parallel_state

            state = get_parallel_state()
            sp_size = int(state.sp_size)
            if sp_size < 1:
                raise RuntimeError(f"MiniMax-H3 SP prompt sharing observed invalid sp_size={sp_size}")
            if sp_size > 1:
                group = state.sp_group
                if group is None:
                    raise RuntimeError(
                        f"MiniMax-H3 SP prompt sharing requires an SP process group for sp_size={sp_size}"
                    )
                group_size = dist.get_world_size(group)
                group_rank = dist.get_rank(group)
                state_rank = int(state.sp_rank)
                if group_size != sp_size or group_rank != state_rank:
                    raise RuntimeError(
                        "MiniMax-H3 SP prompt sharing found an inconsistent VeOmni parallel state: "
                        f"state=(size={sp_size}, rank={state_rank}), "
                        f"group=(size={group_size}, rank={group_rank})"
                    )
        except BaseException as exc:
            state_error = exc

        if not self._sync_embedding_status(state_error is None):
            if state_error is not None:
                raise state_error
            raise RuntimeError("MiniMax-H3 SP prompt sharing setup failed on another rank")

        min_sp_size, max_sp_size = self._sync_embedding_int_bounds(sp_size)
        if min_sp_size != max_sp_size:
            raise RuntimeError(
                "MiniMax-H3 SP prompt sharing requires the same SP size on every role rank, "
                f"observed range [{min_sp_size}, {max_sp_size}]"
            )
        if sp_size <= 1:
            return None
        return group

    def _is_sp_source_rank(self) -> bool:
        """Return whether this rank may run the local conditioner under SP."""
        if not self._share_across_sp:
            return True

        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return True
        try:
            from unirl.train.backend.veomni import _compat

            _compat.ensure_installed()
            from veomni.distributed.parallel_state import get_parallel_state

            state = get_parallel_state()
            return int(state.sp_size) <= 1 or int(state.sp_rank) == 0
        except BaseException:
            return False

    def _resolve_shared(
        self, prompts: Sequence[str], group
    ) -> tuple[dict[str, torch.Tensor], int, int, int, int, bool]:
        """Resolve prompts once per SP group and broadcast their embeddings."""
        import torch.distributed as dist

        group_rank = dist.get_rank(group)
        source = dist.get_global_rank(group, 0)
        is_source = group_rank == 0
        device = torch.device(self.device)

        # Tokenization can itself fail on only one rank. Every rank still sends
        # one fixed-size packet so peers never wait forever in the all-gather.
        token_ids_by_prompt: dict[str, list[int]] = {}
        signature_error: BaseException | None = None
        signature_packet = torch.zeros(33, dtype=torch.uint8, device=device)
        try:
            signature = hashlib.sha256()
            for value in (
                _DISK_CACHE_SCHEMA,
                self._checkpoint_identity,
                str(MINIMAX_H3_TEXT_ENCODER_LAYER),
                str(self._encoder_dtype),
                str(self.dtype),
                str(self._expected_hidden_size),
            ):
                encoded = value.encode("utf-8")
                signature.update(len(encoded).to_bytes(8, "big"))
                signature.update(encoded)
            for prompt in prompts:
                token_ids = self._tokenize(prompt)
                token_ids_by_prompt[prompt] = token_ids
                signature.update(len(token_ids).to_bytes(8, "big"))
                for token_id in token_ids:
                    signature.update(int(token_id).to_bytes(8, "big", signed=True))
            signature_packet[0] = 1
            signature_packet[1:].copy_(torch.tensor(list(signature.digest()), dtype=torch.uint8, device=device))
        except BaseException as exc:
            signature_error = exc

        gathered_signatures = [torch.empty_like(signature_packet) for _ in range(dist.get_world_size(group))]
        dist.all_gather(gathered_signatures, signature_packet, group=group)
        signatures_succeeded = all(bool(value[0].item()) for value in gathered_signatures)
        prompts_match = signatures_succeeded and all(
            torch.equal(value[1:], gathered_signatures[0][1:]) for value in gathered_signatures[1:]
        )
        if not self._sync_embedding_status(prompts_match):
            if signature_error is not None:
                raise signature_error
            if not signatures_succeeded:
                raise RuntimeError("MiniMax-H3 SP prompt signature construction failed on another group rank")
            if not prompts_match:
                raise ValueError("MiniMax-H3 SP prompt sharing requires identical prompt/cache signatures")
            raise RuntimeError("MiniMax-H3 SP prompt sharing failed on another rank or group")

        resolved: dict[str, torch.Tensor] = {}
        memory_hits = disk_hits = misses = 0
        error: BaseException | None = None
        header = torch.zeros(7, dtype=torch.int64, device=device)
        if is_source:
            try:
                resolved, memory_hits, disk_hits, misses = self._resolve_local(prompts)
                shapes = set()
                for prompt in prompts:
                    if prompt not in resolved:
                        raise KeyError(f"MiniMax-H3 SP prompt source did not resolve prompt {prompt!r}")
                    cached = resolved[prompt]
                    self._validate_cached_tensor(cached, token_ids_by_prompt[prompt])
                    shapes.add(tuple(cached.shape))
                if len(shapes) != 1:
                    raise ValueError("MiniMax-H3 SP prompt embeddings must have one common [1, tokens, hidden] shape")
                batch, tokens, hidden = shapes.pop()
                if batch != 1:
                    raise ValueError(f"MiniMax-H3 SP prompt embeddings must have batch size 1, observed {batch}")
                header.copy_(
                    torch.tensor(
                        [1, len(prompts), tokens, hidden, memory_hits, disk_hits, misses],
                        dtype=torch.int64,
                        device=device,
                    )
                )
            except BaseException as exc:
                error = exc

        dist.broadcast(header, src=source, group=group)
        if not self._sync_embedding_status(bool(header[0].item())):
            if error is not None:
                raise error
            raise RuntimeError("MiniMax-H3 SP prompt sharing failed on another rank or group")

        _, unique_count, tokens, hidden, root_memory_hits, root_disk_hits, root_misses = map(int, header.tolist())
        payload = None
        allocation_error: BaseException | None = None
        try:
            payload = self._allocate_shared_payload((unique_count, tokens, hidden), device)
            if is_source:
                payload.copy_(torch.cat([resolved[prompt] for prompt in prompts], dim=0))
        except BaseException as exc:
            allocation_error = exc
        if not self._sync_embedding_status(allocation_error is None):
            if allocation_error is not None:
                raise allocation_error
            raise RuntimeError("MiniMax-H3 SP prompt payload allocation failed on another rank")
        assert payload is not None
        broadcast_error: BaseException | None = None
        try:
            dist.broadcast(payload, src=source, group=group)
        except BaseException as exc:
            broadcast_error = exc
        if not self._sync_embedding_status(broadcast_error is None):
            if broadcast_error is not None:
                raise broadcast_error
            raise RuntimeError("MiniMax-H3 SP prompt payload broadcast failed on another rank or group")
        resolved = {prompt: payload[index : index + 1] for index, prompt in enumerate(prompts)}

        shared_hits = 0 if is_source else len(prompts)
        if not is_source:
            memory_hits = disk_hits = misses = 0
        else:
            memory_hits = root_memory_hits
            disk_hits = root_disk_hits
            misses = root_misses
        return resolved, memory_hits, disk_hits, misses, shared_hits, is_source

    def _allocate_shared_payload(
        self,
        shape: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Allocate the SP broadcast payload in the pipeline output dtype."""
        return torch.empty(shape, dtype=self.dtype, device=device)

    def _sync_embedding_int_bounds(self, value: int) -> tuple[int, int]:
        """Return the world-wide minimum and maximum of one control integer."""
        import torch.distributed as dist

        if dist.get_world_size() <= 1:
            return value, value
        if self._embedding_sync_group is None:
            raise RuntimeError("MiniMax-H3 distributed embedding control group is not initialized")
        bounds = torch.tensor([value, -value], dtype=torch.int64)
        dist.all_reduce(bounds, op=dist.ReduceOp.MIN, group=self._embedding_sync_group)
        return int(bounds[0].item()), -int(bounds[1].item())

    def _resolve_with_disk_cache(self, prompts: Sequence[str], resolved: dict[str, torch.Tensor]) -> tuple[int, int]:
        """Resolve RAM misses from disk and encode only entries absent under lock."""
        entries_by_key: OrderedDict[str, tuple[list[str], list[int], str]] = OrderedDict()
        for prompt in prompts:
            token_ids = self._tokenize(prompt)
            key = self._disk_cache_key(token_ids)
            if key in entries_by_key:
                entries_by_key[key][0].append(prompt)
            else:
                entries_by_key[key] = ([prompt], token_ids, self._disk_cache_path(key))

        disk_hits = 0
        misses = 0
        pending = []
        for entry_prompts, token_ids, path in entries_by_key.values():
            cached = self._load_disk_cache(path, token_ids, warn_on_corruption=False)
            if cached is None:
                pending.append((entry_prompts, token_ids, path))
                continue
            disk_hits += len(entry_prompts)
            for prompt in entry_prompts:
                self._remember(prompt, cached, resolved)
        if not pending:
            return disk_hits, misses

        if self._disk_cache_read_only:
            corrupt = None
            for _, token_ids, path in pending:
                if os.path.exists(path):
                    corrupt = (path, token_ids)
                    break
            if corrupt is not None:
                self._load_disk_cache(corrupt[0], corrupt[1])
            missing_paths = ", ".join(path for _, _, path in pending)
            raise FileNotFoundError(f"MiniMax-H3 read-only prompt embedding cache miss: {missing_paths}")

        cache_dir = self._require_disk_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        with _serialize_node_conditioner_compute(cache_dir, self._checkpoint_identity), ExitStack() as residency:
            residency_entered = False
            encoder_device = self._encoder_device
            for entry_prompts, token_ids, path in sorted(pending, key=lambda entry: entry[2]):
                with self._disk_cache_lock(path):
                    cached = self._load_disk_cache(path, token_ids)
                    if cached is not None:
                        disk_hits += len(entry_prompts)
                        for prompt in entry_prompts:
                            self._remember(prompt, cached, resolved)
                        continue
                    if not residency_entered:
                        residency.enter_context(self._embedding_residency())
                        residency_entered = True
                        encoder_device = self._encoder_device
                    cached = (
                        self._encode_prompt(entry_prompts[0], encoder_device, token_ids=token_ids)
                        .detach()
                        .to("cpu")
                        .contiguous()
                    )
                    self._validate_cached_tensor(cached, token_ids)
                    self._write_disk_cache(path, cached)
                    misses += 1
                    for prompt in entry_prompts:
                        self._remember(prompt, cached, resolved)
        return disk_hits, misses

    def _encode_missing(self, prompts: Sequence[str], resolved: dict[str, torch.Tensor]) -> None:
        """Encode missing prompts together under one conditioner residency window."""
        with self._conditioner_lock, self._embedding_residency():
            encoder_device = self._encoder_device
            for prompt in prompts:
                cached = self._encode_prompt(prompt, encoder_device).detach().to("cpu").contiguous()
                self._remember(prompt, cached, resolved)

    def _remember(
        self,
        prompt: str,
        cached: torch.Tensor,
        resolved: dict[str, torch.Tensor],
    ) -> None:
        """Store one CPU embedding in the resolved batch and rank-local LRU."""
        resolved[prompt] = cached
        self._cache[prompt] = cached
        self._cache.move_to_end(prompt)
        if len(self._cache) > _PROMPT_CACHE_SIZE:
            self._cache.popitem(last=False)

    def _take_prefetched(self, prompts: Sequence[str], resolved: dict[str, torch.Tensor]) -> int:
        """Consume matching prefetched embeddings and validate their identities."""
        if self._prefetch is None:
            return 0
        hits = 0
        entries_by_key: OrderedDict[str, tuple[list[str], list[int]]] = OrderedDict()
        for prompt in prompts:
            token_ids = self._tokenize(prompt)
            key = self._prefetch_key(token_ids)
            if key in entries_by_key:
                entries_by_key[key][0].append(prompt)
            else:
                entries_by_key[key] = ([prompt], token_ids)
        for key, (entry_prompts, token_ids) in entries_by_key.items():
            try:
                found, prefetched = self._prefetch.take(key)
            except PrefetchCancelledError:
                continue
            if not found:
                continue
            require(prefetched is not None, f"MiniMax-H3 prompt prefetch {key} returned no value")
            require(prefetched.digest == key, "MiniMax-H3 prompt prefetch digest mismatch")
            require(prefetched.token_ids == tuple(token_ids), "MiniMax-H3 prompt prefetch token IDs mismatch")
            self._validate_cached_tensor(prefetched.embedding, token_ids)
            for prompt in entry_prompts:
                self._remember(prompt, prefetched.embedding, resolved)
            hits += len(entry_prompts)
        return hits

    @torch.no_grad()
    def _encode_prefetch(self, prompt: str, token_ids: list[int], digest: str) -> _PrefetchedEmbedding:
        """Encode one prompt on the resident CPU conditioner."""
        with self._conditioner_lock:
            encoder_device = self._encoder_device
            require(encoder_device.type == "cpu", "MiniMax-H3 prompt prefetch conditioner moved off CPU")
            embedding = self._encode_prompt(prompt, encoder_device, token_ids=token_ids).detach().to("cpu").contiguous()
        self._validate_cached_tensor(embedding, token_ids)
        return _PrefetchedEmbedding(digest=digest, token_ids=tuple(token_ids), embedding=embedding)

    def _tokenize(self, prompt: str) -> list[int]:
        """Return the input IDs that define one cache entry."""
        return list(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

    def _prefetch_key(self, token_ids: Sequence[int]) -> str:
        """Hash the full identity required to consume one prefetched result."""
        digest = hashlib.sha256()
        for value in (
            _PREFETCH_SCHEMA,
            self._checkpoint_identity,
            str(MINIMAX_H3_TEXT_ENCODER_LAYER),
            str(self._encoder_dtype),
            str(self.dtype),
            str(self._expected_hidden_size),
        ):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        digest.update(len(token_ids).to_bytes(8, "big"))
        for token_id in token_ids:
            digest.update(int(token_id).to_bytes(8, "big", signed=True))
        return digest.hexdigest()

    def _disk_cache_key(self, token_ids: Sequence[int]) -> str:
        """Hash checkpoint, layer, dtype, and tokenizer input IDs."""
        digest = hashlib.sha256()
        for value in (
            _DISK_CACHE_SCHEMA,
            self._checkpoint_identity,
            str(MINIMAX_H3_TEXT_ENCODER_LAYER),
            str(self._encoder_dtype),
        ):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        digest.update(len(token_ids).to_bytes(8, "big"))
        for token_id in token_ids:
            digest.update(int(token_id).to_bytes(8, "big", signed=True))
        return digest.hexdigest()

    @property
    def _encoder_dtype(self) -> torch.dtype:
        return self._text_encoder_dtype

    def _resolve_hidden_size(self) -> int | None:
        """Return the conditioner's expected hidden width when configured."""
        config = getattr(self.text_encoder, "config", None)
        text_config = getattr(config, "text_config", None)
        for source in (text_config, config):
            hidden_size = getattr(source, "hidden_size", None)
            if hidden_size is not None:
                return int(hidden_size)
        return None

    def _disk_cache_path(self, key: str) -> str:
        """Return a sharded cache path for one prompt key."""
        return os.path.join(self._require_disk_cache_dir(), key[:2], f"{key}.pt")

    def _require_disk_cache_dir(self) -> str:
        """Return the configured disk cache directory."""
        require(self._disk_cache_dir is not None, "MiniMax-H3 prompt embedding cache directory is not configured")
        return self._disk_cache_dir

    def _load_disk_cache(
        self,
        path: str,
        token_ids: Sequence[int],
        *,
        warn_on_corruption: bool = True,
    ) -> torch.Tensor | None:
        """Load and validate one CPU cache tensor."""
        try:
            cached = torch.load(path, map_location="cpu", weights_only=True)
            self._validate_cached_tensor(cached, token_ids)
            return cached.contiguous()
        except FileNotFoundError:
            return None
        except Exception as exc:
            if self._disk_cache_read_only:
                raise RuntimeError(f"MiniMax-H3 prompt embedding cache entry is unreadable: {path}") from exc
            if warn_on_corruption:
                logger.warning("Ignoring corrupt MiniMax-H3 prompt embedding cache entry %s: %s", path, exc)
            return None

    def _validate_cached_tensor(self, cached: object, token_ids: Sequence[int]) -> None:
        """Validate the shape, placement, and dtype of one cached embedding."""
        require(isinstance(cached, torch.Tensor), "cached value is not a tensor")
        require(cached.device.type == "cpu", f"cached tensor is on {cached.device}, expected CPU")
        require(
            cached.dtype == self._encoder_dtype,
            f"cached tensor dtype is {cached.dtype}, expected {self._encoder_dtype}",
        )
        require(bool(torch.isfinite(cached).all()), "cached tensor contains non-finite values")
        require(cached.ndim == 3, f"cached tensor has shape {tuple(cached.shape)}, expected [1, tokens, hidden]")
        require(cached.shape[0] == 1, f"cached tensor has batch size {cached.shape[0]}, expected 1")
        require(
            cached.shape[1] == len(token_ids),
            f"cached tensor has {cached.shape[1]} tokens, expected {len(token_ids)}",
        )
        require(cached.shape[2] > 0, "cached tensor has an empty hidden dimension")
        if self._expected_hidden_size is not None:
            require(
                cached.shape[2] == self._expected_hidden_size,
                f"cached tensor hidden size is {cached.shape[2]}, expected {self._expected_hidden_size}",
            )
        require(cached.is_contiguous(), "cached tensor is not contiguous")

    @contextmanager
    def _disk_cache_lock(self, path: str) -> Iterator[None]:
        """Hold the exclusive writer lock for one cache key."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(f"{path}.lock", "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _write_disk_cache(self, path: str, cached: torch.Tensor) -> None:
        """Atomically publish one CPU embedding cache entry."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "wb") as tmp_file:
                torch.save(cached, tmp_file)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

    def _ensure_embedding_sync_group(self) -> None:
        """Build the CPU control group while the ranks are still in lockstep."""
        if self._embedding_sync_group is not None or (
            not self._onload_for_embed
            and self._disk_cache_dir is None
            and not self._share_across_sp
            and not self.prefetch_enabled
        ):
            return
        import torch.distributed as dist

        # new_group is itself a collective over the default group, so it cannot
        # wait until a rank has already spent minutes in an uncached conditioner
        # forward. The barrier spans the whole world rather than one shard group:
        # an onload stalls a whole node, and under HSDP the replica all-reduce
        # crosses nodes just as the shard reduce-scatter stays inside one.
        # Its own group, not the memoized init_gloo_group(): an async DCP save
        # hands that one to a background thread and keeps it, and two threads
        # driving one process group concurrently is undefined.
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            self._embedding_sync_group = dist.new_group(backend="gloo", timeout=_EMBED_SYNC_TIMEOUT)

    def _synchronize_embedding_ranks(self) -> None:
        """Keep delayed conditioner loads out of the next FSDP NCCL collective."""
        if self._embedding_sync_group is None or not self._onload_for_embed:
            return
        import torch.distributed as dist

        dist.monitored_barrier(
            group=self._embedding_sync_group,
            timeout=_EMBED_SYNC_TIMEOUT,
            wait_all_ranks=True,
        )

    def _sync_embedding_status(self, local_ok: bool) -> bool:
        """Propagate prompt failures before a peer enters a later collective."""
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
            return local_ok
        if self._embedding_sync_group is None:
            raise RuntimeError("MiniMax-H3 distributed embedding control group is not initialized")
        status = torch.tensor([int(local_ok)], dtype=torch.int32)
        dist.all_reduce(status, op=dist.ReduceOp.MIN, group=self._embedding_sync_group)
        return bool(status.item())

    def _needs_embedding_status_sync(self) -> bool:
        """Return whether this call participates in a role-wide status collective."""
        import torch.distributed as dist

        return (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
            and self._embedding_sync_group is not None
        )

    def _encode_prompt(
        self,
        prompt: str,
        encoder_device: torch.device,
        *,
        token_ids: list[int] | None = None,
    ) -> torch.Tensor:
        """Run the frozen Qwen3-VL conditioner once for one prompt."""
        token_ids = self._tokenize(prompt) if token_ids is None else token_ids
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=encoder_device)
        # Qwen3-VL lays its 3D rotary positions out per modality run, read
        # off the token type ids the processor derives (0 text, 1 image,
        # 2 video). Text-only here, but the conditioner still wants them.
        mm_token_type_ids = torch.tensor(
            self.processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=encoder_device
        )
        outputs = self.text_encoder.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        return outputs.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER]

    @contextmanager
    def _embedding_residency(self) -> Iterator[None]:
        """Temporarily exchange GPU-resident VAEs for the frozen text encoder."""
        target_device = torch.device(self.device)
        encoder_device = self._encoder_device
        if not self._onload_for_embed or target_device.type != "cuda" or encoder_device.type != "cpu":
            yield
            return

        # Eight DP workers share one node's host-memory channels. Concurrently
        # paging and copying eight 64 GB encoders made every H2D transfer
        # hour-scale; serialize the residency window per node while allowing
        # the two nodes to proceed independently.
        # Each restore is registered BEFORE its move: ``Module.to`` walks
        # parameters in place, so an OOM part-way through the 64 GB onload
        # leaves the conditioner half-resident and still has to be undone.
        with _serialize_node_residency(), ExitStack() as cleanup:
            for module in (self.vae, self.audio_vae):
                device = next(module.parameters()).device
                if device.type == "cuda":
                    cleanup.callback(module.to, device)
                    module.to("cpu")
            torch.cuda.empty_cache()
            cleanup.callback(torch.cuda.empty_cache)
            cleanup.callback(self.text_encoder.to, "cpu")
            self.text_encoder.to(target_device)
            yield


__all__ = ["MiniMaxH3TextEmbedStage"]
