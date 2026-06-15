"""Fused vectorized scatter -- the orchestrator (plan C2).

Single concern: compose the batched body load + expand + token scatter
into the model-facing ``[B, L]`` token tensor, reading each emitted
node's ``_data.bin`` record ONCE (one gather) and assembling with
batched numpy at the geometry prepass's precomputed BFS columns.

Pipeline (each step a single-concern submodule):

1. :func:`._locator.node_token_spans` -- emission nodes -> ``_data.bin``
   token-region spans (header decode only).
2. :func:`._body_load.gather_node_bodies` -- ONE vectorized gather of
   every node's raw u16 token stream into a flat CSR array.
3. :func:`._expand.expand_node_bodies` -- per-node post-promotion /
   strip / shift expansion (owned ``expand_tokens`` semantics) +
   self-token, flattened CSR.
4. :func:`._prefix_values.variant_prefix_values` -- per-row variant
   prefix ids (shifted), body-free from ``_variants.bin``.
5. :func:`._token_scatter.scatter_tokens` -- ONE batched scatter into
   ``tokens[B, L]`` at the geometry columns, with the straddler cut.

The geometry prepass (plan C1) already removed the per-edge metadata
re-parse; this scatter removes the per-edge BODY re-read (one gather)
and the per-row token-assembly Python loop (one vectorized scatter).
The dense identity / numeric sidecars are layered on top in a sibling
pass; this module owns the token-tensor concern.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections

from .._types import BatchGeometry
from ._body_load import gather_node_bodies
from ._expand import ExpandedBatch, expand_node_bodies
from ._locator import node_token_spans
from ._prefix_values import variant_prefix_values
from ._token_scatter import scatter_tokens


__all__ = ["ScatteredTokens", "scatter_batch_tokens"]


@dataclass(frozen=True)
class ScatteredTokens:
    """The token-tensor scatter result.

    ``tokens`` is the model-facing ``u16[B, L]`` tensor (``id == 0`` is
    the null-content pad). ``expanded`` is the per-emitted-node expansion
    (the flat stream + CSR + the threaded per-node state + promotion
    masks) -- exposed so the dense-sidecar pass reuses the SAME single
    body load + expand rather than re-reading bodies (no re-parse in the
    call chain).
    """

    tokens: np.ndarray
    expanded: ExpandedBatch


def scatter_batch_tokens(
    geometry: BatchGeometry,
    *,
    cols: ColumnarSections,
    data_u8: np.ndarray,
    variants_u8: np.ndarray,
) -> ScatteredTokens:
    """Assemble ``tokens[B, L]`` from the geometry + the session bytes.

    Parameters
    ----------
    geometry:
        The body-free prepass result (plan C1).
    cols:
        The columnar ``sections.bin`` catalog (the node locator + the
        variant-prefix vkeys).
    data_u8:
        The arm's ``_data.bin`` as a 1-D ``uint8`` array (read-only
        memmap) -- the ONLY body source, read once per emitted node.
    variants_u8:
        ``_variants.bin`` as a 1-D ``uint8`` array (the variant-prefix
        id source). NOT ``_data.bin``.

    Returns
    -------
    ScatteredTokens
        The ``u16[B, L]`` token tensor + the per-node expanded CSR.
    """
    emission = geometry.emission
    nodes = np.asarray(emission.node, dtype=np.int64)

    starts, counts = node_token_spans(cols, data_u8, nodes)
    bodies = gather_node_bodies(data_u8, starts, counts)
    expanded = expand_node_bodies(bodies, emission.edge_type)

    # The expanded per-node length MUST equal the body-free own_length
    # (1 + realized body_len); a mismatch means the stored geometry and
    # the live body disagree -- surface it rather than mis-scatter.
    expanded_lengths = np.diff(expanded.node_offsets)
    if not np.array_equal(
        expanded_lengths, np.asarray(emission.own_length, dtype=np.int64)
    ):
        raise AssertionError(
            "scatter: expanded per-node length disagrees with the "
            "body-free own_length (1 + realized body_len); the realized-"
            "geometry sidecar and _data.bin are out of sync"
        )

    # Each row's root node carries the variant prefix (the first emitted
    # node of every row). row_offsets[:-1] indexes those roots.
    root_nodes = nodes[emission.row_offsets[:-1]]
    prefix_tokens, prefix_offsets = variant_prefix_values(
        variants_u8, cols, nodes=root_nodes
    )

    tokens = scatter_tokens(
        geometry, expanded, prefix_tokens, prefix_offsets
    )
    return ScatteredTokens(tokens=tokens, expanded=expanded)
