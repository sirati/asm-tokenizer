"""CLI boundary parser for :class:`LengthReduction` (plan ALG-2).

Pure text -> typed conversion at the CLI / config boundary. Once
parsed, downstream code is fully typed and never re-inspects strings.
This module owns the grammar:

- ``"max"`` -> :attr:`ReductionKind.MAX`.
- ``"p1"`` ... ``"p99"`` -> :attr:`ReductionKind.PERCENTILE` with that
  percentile.
- ``"p100"`` -> canonicalises to :attr:`ReductionKind.MAX` (top-of-
  distribution is degenerate-equal to max).

All other inputs (case variants, leading/trailing whitespace, fractional
percentiles, empty string, unknown tokens) are rejected with
:class:`ValueError`.

No imports from ``batch_decode`` here -- this module is upstream of any
batch pipeline; it only needs the typed result class.
"""

from __future__ import annotations

from ._types import LengthReduction, ReductionKind


__all__ = ["parse_reduction"]


def parse_reduction(text: str) -> LengthReduction:
    """Parse a CLI/config reduction spec into a :class:`LengthReduction`.

    Accepts ``"max"`` and ``"p<N>"`` with ``1 <= N <= 100``; ``"p100"``
    canonicalises to :attr:`ReductionKind.MAX`. Case-sensitive: ``"P95"``
    is rejected.
    """
    if text == "max" or text == "p100":
        return LengthReduction(kind=ReductionKind.MAX)
    if text.startswith("p"):
        suffix = text[1:]
        # Reject empty suffix and non-pure-digit suffixes (e.g. "pAB",
        # "p1.5", "p-1", "p 5") so the only accepted forms are
        # ``p<digits>``.
        if not suffix or not suffix.isdigit():
            raise ValueError(f"unparseable reduction: {text!r}")
        n = int(suffix)
        if not (1 <= n <= 99):
            raise ValueError(f"percentile must be 1..100: {n}")
        return LengthReduction(kind=ReductionKind.PERCENTILE, percentile=n)
    raise ValueError(f"unknown reduction: {text!r}")
