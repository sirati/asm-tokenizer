"""Top-level minimum-variant emission gate for the sorted index.

Single concern: decide whether a matched section is EMITTED into the
sorted index based on its top-level (depth-0, root-variant) variant
counts. The gate owns nothing about lengths, reductions, depth, or the
walk -- it answers one question per section: does this section's
top-level variant population clear the configured minimums?

Boundary contract (the design-first sentence):

  *Given the configured minimums + one section's top-level total and
  unique variant counts, return whether the section is emitted. A gated
  -out section is stamped with length 0 -- the same representation a
  0-variant section already takes -- so the wire format, filename
  scheme, and reader are untouched (a length-0 bucket is never drawn at
  any real training target length).*

Composition of the two minimums (per the feature brief):

* ``min_variants`` (N): require at least N top-level variants
  (duplicates included).
* ``min_variants_unique`` (M): require at least M UNIQUE top-level
  variants (after dedup by data-bin pointer). Legal without
  ``--adjust-for-duplicates`` -- gating by uniqueness does not by
  itself change reduction semantics.

When both are set, ``M <= N`` is required (an unsatisfiable gate
``M > N`` is a configuration error, rejected at construction).
"""

from __future__ import annotations

from dataclasses import dataclass


__all__ = ["VariantGate"]


@dataclass(frozen=True)
class VariantGate:
    """Typed top-level minimum-variant gate (off by default).

    ``min_variants`` and ``min_variants_unique`` default to ``0`` which
    disables the respective check (every section clears a ``0``
    threshold). The gate is depth- and mode-independent: it inspects
    only the section's own (depth-0) variants, so a gated-out section is
    excluded from EVERY ``(mode, depth)`` output.

    Construction validation (:meth:`__post_init__`):

    * both thresholds must be ``>= 0``;
    * when both are set (``> 0``), ``min_variants_unique <=
      min_variants`` -- otherwise the gate can never pass and the
      configuration is rejected.
    """

    min_variants: int = 0
    min_variants_unique: int = 0

    def __post_init__(self) -> None:
        if self.min_variants < 0:
            raise ValueError(
                f"min_variants must be >= 0; got {self.min_variants!r}"
            )
        if self.min_variants_unique < 0:
            raise ValueError(
                "min_variants_unique must be >= 0; got "
                f"{self.min_variants_unique!r}"
            )
        if (
            self.min_variants > 0
            and self.min_variants_unique > 0
            and self.min_variants_unique > self.min_variants
        ):
            raise ValueError(
                "min_variants_unique must be <= min_variants; got "
                f"min_variants_unique={self.min_variants_unique!r} > "
                f"min_variants={self.min_variants!r} (unsatisfiable gate)"
            )

    def passes(self, *, n_total: int, n_unique: int) -> bool:
        """Whether a section with these top-level counts is emitted.

        ``n_total`` is the section's top-level variant count (duplicates
        included); ``n_unique`` is the count of distinct data-bin
        pointers among them. A disabled threshold (``0``) is cleared by
        any count.
        """
        return n_total >= self.min_variants and n_unique >= self.min_variants_unique
