"""The build's no-cutoff length budget constant.

Single concern: the no-cutoff budget the per-root BFS (:mod:`._bfs`)
asserts every spliced length stays under. The per-variant OWN body
length the BFS sums is no longer parsed here -- it is the matched-arm
realized-length sidecar (:mod:`tokenizer.aligned_data.realized_lengths`),
generated as its own Phase-4a pass and INJECTED into the BFS by the
builder, so the index build never re-decodes ``_data.bin`` geometry.
The splice adjacency (which callees a node reaches) lives in
:mod:`._adjacency`, read live from the catalog memmap with no
precomputed graph structure.
"""

from __future__ import annotations


__all__ = ["LARGE_CONTEXT_LEN"]


#: Same role as the historical walk budget: the build ASSERTS every
#: spliced length stays under this bound, because beyond it the legacy
#: stage-2 cutoff would have fired and the sorted index would silently
#: under-report (plan D-2.2).
LARGE_CONTEXT_LEN = 2**30
