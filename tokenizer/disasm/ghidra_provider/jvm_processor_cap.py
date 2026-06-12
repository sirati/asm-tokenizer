"""Per-worker Ghidra JVM processor-cap sizing.

Single concern: own the answer to "how many processors should THIS
worker's Ghidra JVM size its thread pools from, and which JVM flags
enforce that?".

Why this module exists
----------------------
Each tokenize worker boots one process-global Ghidra JVM (via pyghidra).
Ghidra — and the JVM itself (ForkJoinPool, GC threads, …) — size their
thread pools from ``Runtime.getRuntime().availableProcessors()``, which
inside a container reports ALL host cores. With ``N`` workers pinned to
one node (the dispatch's per-machine worker count), an unscoped JVM gives
every worker the full machine width, so a node runs ``N`` machine-wide
Ghidra instances → ``N×`` CPU oversubscription that starves the
coordinator loop.

Capping each worker at ``ceil(machine_cores / workers_per_node)``
processors keeps the aggregate thread demand at ~one machine width.

Flags emitted
-------------
* ``-XX:ActiveProcessorCount=<K>`` caps ``availableProcessors()`` itself,
  so EVERY derived pool (Ghidra's analysis pools, ForkJoinPool.common,
  the GC, …) sees ``K``.
* ``-Dcpu.core.limit=<K>`` is Ghidra's own documented launch property
  (``support/launch.properties``): Ghidra uses ``min(availableProcessors,
  K)`` for its analysis thread pools. Redundant with the JVM cap above
  but belt-and-braces — it pins Ghidra's own knob to the same ``K`` so a
  future JVM that ignored ``ActiveProcessorCount`` would still be bounded.

Both honour the SAME ``K`` and both apply ``min``-against-detected
semantics, so they compose without conflict.

This module is pure: it computes ``K`` and renders the flag strings. The
provider's JVM-startup site (``provider._ensure_jvm_started``) consumes
the rendered flags; the worker entry point computes ``K`` from its
``--cores`` arg and registers it. Neither caller needs to know the flag
spellings — that knowledge lives here alone.
"""

from __future__ import annotations

import math


def compute_processor_cap(machine_cores: int, workers_per_node: int) -> int:
    """Return the per-worker processor cap = ``ceil(machine_cores /
    workers_per_node)``, floored at 2.

    Ceiling division (not floor) so the aggregate cap across all workers
    is at least the machine width — under-provisioning every worker would
    leave cores idle.

    The floor is 2, not 1: ``-XX:ActiveProcessorCount=1`` puts the whole
    JVM into single-processor mode — SerialGC, no parallel JIT, every
    derived pool at size 1 — which costs far more wall-clock per task
    than the one extra logical thread of oversubscription it avoids
    (observed on 14-core/14-worker LMU nodes, where the quotient is 1).
    The worker-side nice offset already bounds the contention damage of
    the resulting <=2x logical oversubscription.

    Raises ``ValueError`` on non-positive inputs: a caller that cannot
    determine a real ``workers_per_node`` must fall back to NO cap rather
    than feed a sentinel here (see ``provider`` / the worker entry).
    """
    if machine_cores <= 0:
        raise ValueError(f"machine_cores must be positive, got {machine_cores}")
    if workers_per_node <= 0:
        raise ValueError(
            f"workers_per_node must be positive, got {workers_per_node}"
        )
    return max(2, math.ceil(machine_cores / workers_per_node))


def processor_cap_vmargs(cap: int) -> tuple[str, ...]:
    """Render the JVM flags that pin Ghidra's thread-pool sizing to
    ``cap`` processors.

    Returns the ``-XX:ActiveProcessorCount`` JVM cap plus Ghidra's
    ``-Dcpu.core.limit`` launch property, both carrying the same ``cap``.
    See the module docstring for why both are emitted.
    """
    return (
        f"-XX:ActiveProcessorCount={cap}",
        f"-Dcpu.core.limit={cap}",
    )
