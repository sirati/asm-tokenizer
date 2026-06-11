"""Parse a per-binary ``fullname`` into compilation axes.

Single concern: turn a fullname prefix such as
``armv7l-hf-clang-10.0.1-Oz_hello__15f3f338`` into the
``(isa, comp, comp_version, optim, program)`` tuple plus the derived
``isa_family`` / ``bitness``.  No filesystem, no DB.

The fullname shape is ``<isa>-<comp>-<compv>-<optim>_<program>``:

* The ISA is recognised FIRST, as the longest matching prefix from
  :func:`tokenizer.arch_translation.all_known_arch_strings` — exactly
  the disambiguation strategy already used by
  ``tokenizer.vocab_unifier.loader.load_vocab_manager``.
* ``program`` = everything after the first underscore **of the
  post-ISA remainder** (per the documented "after the first underscore"
  rule, but applied after the ISA is stripped).  It may itself contain
  ``__`` and a hash suffix, e.g. ``hello__15f3f338``.

Matching the ISA before splitting the program matters because the ISA
token can itself contain an underscore (``x86_64``); splitting the raw
fullname on its first underscore would sever ``x86_64`` into ``x86`` +
``64-...`` and mis-parse every x86-64 binary.  The ISA token can also
contain dashes (``armv7l-hf``, ``ppc64le`` ...), which the longest-prefix
match likewise absorbs.

After the ISA is stripped, the remaining ``<comp>-<compv>-<optim>`` is
parsed from both ends: ``comp`` is the first dash field, ``optim`` is the
last, and ``comp_version`` is everything in between (so a dashed version
survives).

``isa_family`` and ``bitness`` are derived from the compact
``Platform`` that :func:`tokenizer.arch_translation.arch_to_platform`
maps the ISA onto — the single source of truth for arch normalisation.

Unparseable fullnames yield an :class:`Axes` whose axis fields are all
``None`` (``program`` is still filled from the first-underscore split
when present); callers record the row with NULL axes and a warning
rather than crashing.
"""

from __future__ import annotations

from dataclasses import dataclass

from tokenizer.arch import Platform
from tokenizer.arch_translation import all_known_arch_strings, arch_to_platform

# Compact ``Platform`` literal → (family, bitness).  ``Platform`` is the
# bitness-only normalised arch enum; this collapses it to the coarse
# family used by the ratio queries and recovers the bitness integer.
# x86 family covers x86 (32) and x64 (64) per the task spec.
_PLATFORM_TO_FAMILY_BITNESS: dict[Platform, tuple[str, int]] = {
    "x86": ("x86", 32),
    "x64": ("x86", 64),
    "arm32": ("arm", 32),
    "arm64": ("arm", 64),
    "mips32": ("mips", 32),
    "mips64": ("mips", 64),
    "ppc32": ("ppc", 32),
    "ppc64": ("ppc", 64),
    "riscv32": ("riscv", 32),
    "riscv64": ("riscv", 64),
}

# Longest-first so ``armv7l-hf`` wins over ``arm`` and ``mips64el`` over
# ``mips`` — without this a shorter alias would shadow a longer one.
_KNOWN_ISAS: tuple[str, ...] = tuple(
    sorted(all_known_arch_strings(), key=len, reverse=True)
)


@dataclass(frozen=True, slots=True)
class Axes:
    """The compilation axes parsed from a fullname.

    Every axis field is ``None`` when the fullname does not parse;
    ``program`` is still populated from the first-underscore split when
    one is present.  ``isa_exact`` is the literal ISA token as it
    appears in the fullname (``armv7l-hf``); ``isa_family`` / ``bitness``
    are derived from the normalised ``Platform``.
    """

    program: str | None
    isa_exact: str | None
    isa_family: str | None
    bitness: int | None
    comp: str | None
    comp_version: str | None
    optim_level: str | None

    @property
    def parsed(self) -> bool:
        """True when the full axis tuple was recovered."""
        return self.isa_exact is not None


def _program_after_first_underscore(text: str) -> str | None:
    """Everything after the first underscore in ``text`` (or ``None``
    when there is no underscore)."""
    _head, sep, program = text.partition("_")
    return program if sep else None


def _match_isa(fullname: str) -> tuple[str, str] | None:
    """Return ``(isa_exact, remainder)`` for the longest known-ISA
    prefix of ``fullname`` (remainder is what follows ``<isa>-``), or
    ``None`` when no known ISA prefixes the string.

    The match is against the **whole fullname**, not a pre-split prefix,
    because an ISA token may itself contain an underscore (``x86_64``);
    splitting on the first underscore first would sever it.  The program
    is therefore split out of the remainder (post-``<isa>-``), not the
    raw fullname."""
    for isa in _KNOWN_ISAS:
        if fullname.startswith(isa + "-"):
            return (isa, fullname[len(isa) + 1 :])
    return None


def _family_bitness(isa_exact: str) -> tuple[str | None, int | None]:
    """Derive ``(family, bitness)`` from an ISA token via the normalised
    ``Platform``; ``(None, None)`` when the ISA is not catalogued."""
    try:
        platform = arch_to_platform(isa_exact)
    except ValueError:
        return (None, None)
    return _PLATFORM_TO_FAMILY_BITNESS.get(platform, (None, None))


def parse_axes(fullname: str) -> Axes:
    """Parse ``fullname`` into its compilation :class:`Axes`.

    Never raises: an unrecognised ISA prefix or a comp/optim tuple with
    fewer than three dash fields yields an :class:`Axes` with NULL axis
    fields (``program`` still filled when present)."""
    isa_match = _match_isa(fullname)
    if isa_match is None:
        # No known ISA prefix — recover the program from the raw
        # first-underscore split and leave the axes NULL.
        return Axes(
            _program_after_first_underscore(fullname),
            None, None, None, None, None, None,
        )
    isa_exact, remainder = isa_match

    # remainder is ``<comp>-<compv>-<optim>_<program>``; the program is
    # split AFTER the ISA so an underscore inside the ISA (``x86_64``)
    # cannot sever the prefix.
    axis_tail, sep, program_field = remainder.partition("_")
    program = program_field if sep else None

    # axis_tail is ``<comp>-<compv>-<optim>``; parse from both ends so a
    # dashed comp_version (the middle) survives intact.
    fields = axis_tail.split("-")
    if len(fields) < 3:
        return Axes(program, None, None, None, None, None, None)
    comp = fields[0]
    optim = fields[-1]
    comp_version = "-".join(fields[1:-1])

    family, bitness = _family_bitness(isa_exact)
    return Axes(
        program=program,
        isa_exact=isa_exact,
        isa_family=family,
        bitness=bitness,
        comp=comp,
        comp_version=comp_version,
        optim_level=optim,
    )
