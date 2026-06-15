"""Sorted-index graph-lengths: index-build invariants (no loader parity).

RETIRED: the loader<->index length-PARITY gate.

This file used to pin ``compute_node_lengths`` (the FULL-variant-set
sorted-index build) against the dataloader callee walk
(:func:`walk_callees` + :func:`expand_tokens`) per (section, variant,
depth), asserting EQUALITY. That equality held only because both
consumers ran their once-only inclusion over the section's FULL variant
set.

The loader now runs its once-only inclusion over the SAMPLED variant
subset (the user's Decision 4: the columnwise-ALL exclusion is a
property of the rows that actually emit). The sorted-index + realized-
length builds DELIBERATELY stay FULL-SET, so the index length is now a
LOOSE UPPER BOUND on the emitted token count, not an equality (the
index/backfill reconciles the slack). The cross-consumer parity is
therefore intentionally false and has been retired -- it is NOT weakened
to ``<=``.

The single-source semantic guarantee that survives: both consumers still
drive the SAME :class:`OnceOnlyInclusion` decider, so the inclusion
ALGORITHM cannot drift -- only the variant set each feeds it differs.
The index build's own once-only / exclusion semantics are pinned by the
byte-identity golden (``test_build_golden.py``) + the variant-count /
shape / budget-guard coverage in ``test_length_compute.py``; the loader
subset semantics are pinned in
``..loader.batch_decode.tests.test_callee_walk``.

What remains here: the index-build invariants that never referenced the
loader oracle (e.g. depth-0 = body + 1, no splice).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tokenizer.aligned_data.csv_section_index import (
    read_csv_section_index_arrays,
)
from tokenizer.aligned_data.loader.tests._corpus import (
    MatchedFunctionSpec,
    VariantSpec,
    build_corpus,
    make_simple_variant,
)
from tokenizer.aligned_data.matched_sections_columnar import (
    parse_sections_columnar,
)
from tokenizer.aligned_data.sorted_index._graph_lengths import (
    compute_node_lengths,
)
from tokenizer.aligned_data.sorted_index.tests._length_helpers import (
    reference_body_lengths,
)


def _shared(n: int, *, seed: int) -> tuple:
    """``n`` variants under the SHARED vkey namespace ``("V", i)``."""
    return tuple(
        make_simple_variant(("V", i), token_seed=seed + i, n_tokens=6 + i)
        for i in range(n)
    )


def _callset(seed: int, per_variant_called) -> tuple:
    """Shared-vkey variants with PER-VARIANT call sets."""
    out = []
    for i, called in enumerate(per_variant_called):
        base = make_simple_variant(("V", i), token_seed=seed + i, n_tokens=6 + i)
        out.append(
            VariantSpec(
                vkey=base.vkey,
                tokens=base.tokens,
                block_rl=base.block_rl,
                insn_rl=base.insn_rl,
                called=tuple(called),
            )
        )
    return tuple(out)


def test_depth0_equals_own_body_plus_one(tmp_path: Path) -> None:
    # Depth-0 spliced length is byte-identical to the legacy build:
    # 1 self-token + the contributing body length per variant (the
    # injected sidecar body length), with NO splice contribution. Pins
    # that the BFS composes the self-token at the DP site (own = body + 1)
    # and that depth-0 carries no splice. Independent of the loader walk.
    specs = [
        MatchedFunctionSpec(
            func_name="root", variants=_callset(1, [("leaf",), ()]),
            called=("leaf",),
        ),
        MatchedFunctionSpec(
            func_name="leaf", variants=_shared(2, seed=21), called=(),
        ),
    ]
    build_corpus(tmp_path, "d0id", matched=specs)
    starts, _l = read_csv_section_index_arrays(tmp_path / "d0id_index.bin")
    blob = np.fromfile(tmp_path / "d0id_sections.bin", dtype=np.uint8)
    data = np.fromfile(tmp_path / "d0id_data.bin", dtype=np.uint8)
    cols = parse_sections_columnar(blob, starts, _l)
    body = reference_body_lengths(cols, data)
    got = compute_node_lengths(cols, starts, body, [0])
    expected = body + 1
    np.testing.assert_array_equal(got[0], expected)
