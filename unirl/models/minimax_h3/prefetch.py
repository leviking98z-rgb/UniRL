"""Bounded single-worker prefetch protocol for MiniMax-H3 prompt embeddings."""

from __future__ import annotations

import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable, Generic, Hashable, Optional, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class PrefetchCancelledError(RuntimeError):
    """Raised when shutdown cancels queued prefetch work."""


@dataclass
class _Entry(Generic[K, V]):
    """One queued, running, or completed prefetch."""

    key: K
    producer: Callable[[], V]
    done: threading.Event = field(default_factory=threading.Event)
    value: Optional[V] = None
    error: Optional[BaseException] = None
    claimed: bool = False


class BoundedPrefetcher(Generic[K, V]):
    """Run keyed producers on one background thread with bounded retained state."""

    def __init__(self, capacity: int, *, thread_name: str) -> None:
        if capacity < 1:
            raise ValueError(f"prefetch capacity must be >= 1, got {capacity}")
        self.capacity = int(capacity)
        self._thread_name = thread_name
        self._condition = threading.Condition()
        self._entries: "OrderedDict[K, _Entry[K, V]]" = OrderedDict()
        self._queue: "deque[_Entry[K, V]]" = deque()
        self._thread: Optional[threading.Thread] = None
        self._active: Optional[_Entry[K, V]] = None
        self._closed = False

    @property
    def retained(self) -> int:
        """Return the number of queued, running, or completed entries."""
        with self._condition:
            return len(self._entries)

    @property
    def closed(self) -> bool:
        """Return whether shutdown has rejected future submissions."""
        with self._condition:
            return self._closed

    def submit(self, key: K, producer: Callable[[], V]) -> bool:
        """Submit one key, returning false when closed or at capacity."""
        with self._condition:
            if self._closed:
                return False
            if key in self._entries:
                return True
            if len(self._entries) >= self.capacity:
                return False
            entry = _Entry(key=key, producer=producer)
            self._entries[key] = entry
            self._queue.append(entry)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name=self._thread_name, daemon=True)
                self._thread.start()
            self._condition.notify()
            return True

    def take(self, key: K) -> tuple[bool, Optional[V]]:
        """Consume a matching entry, blocking until it completes."""
        with self._condition:
            entry = self._entries.get(key)
            if entry is None:
                return False, None
            if entry.claimed:
                raise RuntimeError(f"prefetch entry {key!r} already has a consumer")
            entry.claimed = True

        entry.done.wait()
        with self._condition:
            current = self._entries.get(key)
            if current is not entry:
                raise RuntimeError(f"prefetch entry {key!r} changed before consumption")
            del self._entries[key]
            error = entry.error
            value = entry.value
        if error is not None:
            raise error
        return True, value

    def shutdown(self) -> None:
        """Cancel queued work, finish the active producer, and join the worker."""
        with self._condition:
            if self._closed:
                thread = self._thread
            else:
                self._closed = True
                while self._queue:
                    entry = self._queue.popleft()
                    entry.error = PrefetchCancelledError(f"prefetch entry {entry.key!r} cancelled during shutdown")
                    entry.done.set()
                self._condition.notify_all()
                thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        with self._condition:
            for key, entry in list(self._entries.items()):
                if not entry.claimed and entry is not self._active:
                    del self._entries[key]

    def _run(self) -> None:
        """Drain accepted producers until shutdown."""
        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait()
                if not self._queue:
                    return
                entry = self._queue.popleft()
                self._active = entry
            try:
                entry.value = entry.producer()
            except BaseException as exc:
                entry.error = exc
            finally:
                entry.done.set()
                with self._condition:
                    self._active = None


__all__ = ["BoundedPrefetcher", "PrefetchCancelledError"]
