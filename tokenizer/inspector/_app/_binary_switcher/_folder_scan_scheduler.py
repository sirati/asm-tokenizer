"""Async per-directory loadable-probe scheduler for the folder picker.

Single concern: schedule :func:`is_loadable_for_any` probes for tree
nodes off the UI thread, cache the verdict for the dialog session,
and notify the caller via a callback when a node's loadable flag is
known.

The folder-picker dialog owns the tree widget; this module owns the
probe queue + worker + result caches. The dialog mounts every child
synchronously (with ``loadable=False`` or a cached green) and then
enqueues unknown children here; the worker drains the queue FIFO
(= breadth-first by mount order) and yields every
:data:`_SCAN_BATCH_SIZE` probes so the UI stays responsive on a
directory with hundreds of entries (e.g. ``/tmp``).

Two ``set[Path]`` caches memoise probe results for the dialog
session:

* ``_known_green`` -- paths confirmed loadable. The dialog reads this
  at mount time so re-roots paint cached greens without re-probing.
* ``_fully_checked`` -- folders whose children have all been
  examined. The dialog reads this to skip the worker enqueue entirely
  when the whole child set's verdict is already known: any child not
  in ``_known_green`` is known non-loadable for the dialog session.

Both caches are dialog-session-scoped; :meth:`FolderScanScheduler.shutdown`
clears them so the next dialog opens cold.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, Optional

from textual.widget import Widget
from textual.widgets._tree import TreeNode
from textual.worker import Worker

from ._scan import is_loadable_for_any


__all__ = [
    "FolderScanScheduler",
]


# Number of per-child probes performed between event-loop yields. Small
# enough that even a directory the size of ``/tmp`` (a few hundred
# entries) yields well under the keystroke latency budget; large
# enough that the cooperative-yield overhead stays negligible for
# typical small folders.
_SCAN_BATCH_SIZE = 8


# The result-applied callback fires on the UI thread (worker is an
# async coroutine on the same loop) once a probe lands.
ResultCallback = Callable[[TreeNode, Path, bool], None]


class FolderScanScheduler:
    """Async BFS scheduler for per-folder ``is_loadable_for_any`` probes.

    Lifecycle: instantiate in the dialog's ``__init__``, call
    :meth:`start` from the dialog's ``on_mount`` (the textual worker
    needs an active app), call :meth:`enqueue` from
    ``_populate_folder``, call :meth:`cancel_and_reset` from the
    dialog's ``_rebase`` (drops in-flight probes targeting the
    now-cleared subtree but PRESERVES the path caches), call
    :meth:`shutdown` from the dialog's dismiss path.

    The caller owns the dialog (``host``) so the worker is scoped to
    its lifetime via ``host.run_worker``; cancelling the worker on
    rebase touches only this dialog's probes.
    """

    def __init__(
        self,
        host: Widget,
        on_result: ResultCallback,
    ) -> None:
        """Bind the scheduler to its host widget + result callback.

        Args:
            host: Widget whose ``run_worker`` spawns the drain task.
                Cancellation of the worker is scoped to this widget so
                app-level workers running in parallel are unaffected.
            on_result: Invoked as ``on_result(node, path, loadable)``
                once a probe lands. The callback runs on the UI thread
                (worker is an async coroutine, no thread crossing) so
                it may directly mutate :class:`TreeNode` state.
        """
        self._host = host
        self._on_result = on_result
        # ``asyncio.Queue()`` binds to the running loop on first
        # ``get``, not at construction, so creating it eagerly here is
        # safe even when no loop is yet active (the dialog's compose
        # path enqueues during widget setup, BEFORE ``on_mount`` spawns
        # the worker).
        self._queue: asyncio.Queue[tuple[TreeNode, Path]] = asyncio.Queue()
        # Pending-probe counter per parent folder. Incremented in
        # :meth:`enqueue`, decremented when the worker handles each
        # entry; a zero reading promotes the parent into
        # :attr:`_fully_checked`. Resets in :meth:`cancel_and_reset`
        # because a cancelled worker discards in-flight entries.
        self._pending_per_parent: dict[Path, int] = {}
        self._worker: Optional[Worker[None]] = None
        # Per-dialog-session probe-result caches. The dialog reads
        # these at mount time; the worker writes them as probes land.
        self._known_green: set[Path] = set()
        self._fully_checked: set[Path] = set()

    # --- caches (read-only views for the dialog) -----------------

    def is_known_green(self, path: Path) -> bool:
        """``True`` iff ``path`` has been confirmed loadable in this
        dialog session (worker probe landed or pre-seeded sync probe).
        """
        return path in self._known_green

    def is_fully_checked(self, folder: Path) -> bool:
        """``True`` iff EVERY child of ``folder`` has been probed in
        this dialog session. The dialog uses this to short-circuit the
        enqueue step in :meth:`_populate_folder`: any child not in
        :meth:`is_known_green` is known non-loadable for the session.
        """
        return folder in self._fully_checked

    def remember_green(self, path: Path) -> None:
        """Record ``path`` as confirmed loadable.

        The dialog calls this for a synchronously-probed root (single
        call, not the per-child fan-out) so a subsequent rebase-back
        sees the cached verdict.
        """
        self._known_green.add(path)

    # --- enqueue + lifecycle -------------------------------------

    def enqueue(self, node: TreeNode, path: Path) -> None:
        """Schedule a ``is_loadable_for_any`` probe for ``path``.

        The probe runs FIFO (breadth-first by enqueue order: callers
        enqueue an entire current-level child set before any child
        gets expanded, so current-level verdicts always land before
        grandchildren probes). ``path.parent`` derivation gates the
        :attr:`_fully_checked` promotion when the parent's whole
        child set has been examined.
        """
        parent = path.parent
        self._pending_per_parent[parent] = (
            self._pending_per_parent.get(parent, 0) + 1
        )
        self._queue.put_nowait((node, path))

    def mark_fully_checked(self, folder: Path) -> None:
        """Force-promote ``folder`` into :attr:`_fully_checked`.

        The dialog calls this for folders with no unknown children
        (empty / all-cached-green): the worker counter never fires
        because no entries were enqueued, but the cache benefit
        applies on re-visit.
        """
        self._fully_checked.add(folder)

    def start(self) -> None:
        """Spawn the drain worker bound to ``self._host``'s event
        loop. Called from the dialog's ``on_mount``.

        ``exclusive=True`` scopes cancellation to this scheduler's
        group: a stray older worker (defensive) is cancelled rather
        than racing the queue. Group + name include
        ``id(self)`` so concurrent picker instances do not interfere.
        """
        # Group name carries the scheduler's id so concurrent dialogs
        # (defensive: two pickers cannot normally be open at once but
        # the group scoping costs nothing) do not cancel each other.
        group = f"folder-picker-scan-{id(self)}"
        self._worker = self._host.run_worker(
            self._drain_loop(),
            group=group,
            name=group,
            exclusive=True,
            exit_on_error=False,
        )

    def cancel_and_reset(self) -> None:
        """Cancel the worker + flush the queue + counter dict.

        Called from the dialog's :meth:`_rebase`: in-flight probes
        targeting the now-cleared subtree would otherwise race the
        re-mount + write stale labels onto detached nodes. The path
        caches (:attr:`_known_green`, :attr:`_fully_checked`) are
        PRESERVED so a navigation back to a previously-scanned subtree
        paints cached greens immediately.
        """
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None
        self._queue = asyncio.Queue()
        self._pending_per_parent = {}

    def shutdown(self) -> None:
        """Tear the worker down + drop the session caches.

        Called from the dialog's dismiss path: a long-tail scan against
        a deep ``/tmp`` must not keep running after the user closed the
        picker. The caches are dialog-session-scoped per design (the
        filesystem may have changed by the next dialog), so they reset
        here.
        """
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None
        self._known_green.clear()
        self._fully_checked.clear()
        self._pending_per_parent.clear()

    # --- worker ---------------------------------------------------

    async def _drain_loop(self) -> None:
        """Drain :attr:`_queue` FIFO, yielding every batch.

        Each entry is a ``(node, path)`` pair: the worker probes
        ``path`` via :func:`asyncio.to_thread` (the underlying
        :func:`is_loadable_for_any` is a fully-blocking ``rglob`` walk
        — on a directory like ``/tmp`` a single probe can take seconds,
        so it MUST run off the event loop), updates the caches, and
        calls :attr:`_on_result` on the UI thread once the verdict
        lands. Yields every :data:`_SCAN_BATCH_SIZE` probes so
        keystroke / scroll events run between batches.
        """
        batch = 0
        while True:
            node, path = await self._queue.get()
            try:
                if path in self._known_green:
                    loadable = True
                else:
                    loadable = await asyncio.to_thread(
                        is_loadable_for_any, path
                    )
                    if loadable:
                        self._known_green.add(path)
                self._on_result(node, path, loadable)
                parent = path.parent
                if parent in self._pending_per_parent:
                    self._pending_per_parent[parent] -= 1
                    if self._pending_per_parent[parent] == 0:
                        self._fully_checked.add(parent)
                        del self._pending_per_parent[parent]
            finally:
                self._queue.task_done()
            batch += 1
            if batch >= _SCAN_BATCH_SIZE:
                batch = 0
                # Cooperative yield: hands control back to the event
                # loop so queued UI events (keystrokes, scroll) run
                # between scan batches.
                await asyncio.sleep(0)
