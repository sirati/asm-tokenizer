"""Variable-length function-name generator + mod-4 residue assertion.

Single concern: produce function-name lists whose lengths cycle across
a span wide enough that the resulting section CSV byte offsets cover
every ``mod 4`` residue ``{0, 1, 2, 3}``, NOT only the residue the
legacy ``matched_fn_{i}`` (uniform 12-char) names accidentally hit.

The audit blocker that motivated batch 2 of memoized-booping-wren
(``_pass2.write_matched_sections_pass2`` feeding non-4-aligned CSV
offsets into a 4-byte-asserting writer) was hidden by uniform-length
fixtures whose accidentally-aligned offsets never tripped the
assertion. The residue-coverage assertion below documents the intent
so a future "let's reintroduce alignment on this path" regression
cannot pass CI silently.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence


def make_variable_length_names(
    prefix: str,
    *,
    count: int,
    base_len: int = 7,
    span: int = 8,
) -> List[str]:
    """Return ``count`` distinct names whose lengths cycle through
    ``base_len`` .. ``base_len + span - 1``.

    Default ``base_len=7, span=8`` -> lengths ``7..14`` repeating;
    covering each mod-4 residue twice over a single span and producing
    section CSV starts that span ``{0, 1, 2, 3}`` without any
    coincidence (the *deltas* between successive starts vary, which is
    what drives residue cycling).

    Names follow the form ``f"{prefix}_{filler}{i:03d}"`` where
    ``filler`` is a deterministic stretch of ``x``s sized to bring the
    total length to ``base_len + (i % span)``. The trailing
    ``f"{i:03d}"`` keeps every name unique even when the filler width
    repeats across the span.
    """
    if span < 4:
        raise ValueError("span must be >= 4 to cover all mod-4 residues")
    out: List[str] = []
    for i in range(count):
        target = base_len + (i % span)
        suffix = f"{i:03d}"
        fixed = len(prefix) + 1 + len(suffix)  # prefix + "_" + suffix
        filler_len = max(0, target - fixed)
        filler = "x" * filler_len
        name = f"{prefix}_{filler}{suffix}"
        assert len(name) >= min(target, fixed)
        out.append(name)
    return out


def assert_mod4_residues_covered(
    starts: Iterable[int],
    *,
    expected: Sequence[int] = (0, 1, 2, 3),
) -> None:
    """Intent assertion: ``starts`` must cover every residue in ``expected``.

    Without this guard a future change to
    :func:`make_variable_length_names` (or to the section CSV row
    width) could accidentally regress to the matched_fn_00 pattern
    (only one residue hit) and silently restore the
    alignment-blind-spot the audit caught.
    """
    residues = sorted({int(s) % 4 for s in starts})
    missing = sorted(set(expected) - set(residues))
    assert not missing, (
        f"section starts cover residues {residues}, expected all of "
        f"{sorted(expected)}; missing {missing}. Fixture lengths are no "
        f"longer spanning every mod-4 residue -- adjust "
        f"make_variable_length_names span/base_len."
    )
