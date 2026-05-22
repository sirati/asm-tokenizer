"""Variable-length function-name generator + 4-byte alignment assertion.

Single concern: produce function-name lists whose lengths cycle across
a span wide enough that the section CSV writer's padding policy is
exercised across every input residue (the writer adds 1-4 trailing
``\\n`` bytes after each section so the next section starts on a
4-byte boundary). Without name-length variation the un-padded width
already lands on a boundary every time and the padding code is never
exercised.

After the writer's pad-policy landed, every per-section CSV start is
4-byte aligned by construction; the assertion below pins that
invariant so a future writer regression that drops the padding fails
loudly. The prior "covers every mod-4 residue" assertion came from a
pre-padding world where alignment was deliberately NOT enforced and
the residue spread was the diagnostic.
"""

from __future__ import annotations

from typing import Iterable, List


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


def assert_starts_4_byte_aligned(starts: Iterable[int]) -> None:
    """Pin the post-padding invariant: every CSV-section start is 4-aligned.

    The writer adds 1-4 trailing ``\\n`` bytes after each section so
    the next section header lands on a 4-byte boundary. A regression
    that drops the padding step would land starts on misaligned
    offsets; this assertion fires before the misaligned starts reach
    ``pack_csv_section_index_entry`` (which would also raise but with
    a less obvious message).
    """
    misaligned = sorted({int(s) for s in starts if int(s) % 4 != 0})
    assert not misaligned, (
        f"matched-sections CSV starts must all be 4-byte aligned "
        f"(writer pads with trailing newlines); got misaligned offsets "
        f"{misaligned}"
    )
