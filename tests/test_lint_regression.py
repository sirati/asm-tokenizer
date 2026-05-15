"""Phase J.5 lint regression guards for the owned-API refactor.

Each test is a grep-style file scan over a scoped subset of the
``tokenizer/`` package. The guards encode invariants the refactor
explicitly established (e.g. no Capstone-shape reads in arch
consumers, no magic op-type ints, no eager hot-path materialization).

Lines that start with ``#`` and lines inside triple-quoted strings
are skipped — comment / docstring mentions of legacy names are
acceptable, only live code matches count.
"""

from __future__ import annotations

import re
from pathlib import Path

# ----------------------------------------------------------------------
# Scan primitive
# ----------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
TOKENIZER = REPO_ROOT / "tokenizer"


def _iter_py_files(scope: Path):
    """Yield every ``.py`` file under ``scope`` (file or dir)."""
    if scope.is_file():
        if scope.suffix == ".py":
            yield scope
        return
    for path in sorted(scope.rglob("*.py")):
        yield path


def _code_lines(path: Path):
    """Yield ``(lineno, line)`` for code lines only.

    Skips blank lines, comment-only lines (leading ``#``), and lines
    inside triple-quoted docstrings (toggle on ``\"\"\"``). This is the
    intentionally simple detector the spec calls for — false negatives
    on edge cases like ``'''`` strings or single-line ``\"\"\"...\"\"\"``
    are acceptable.
    """
    in_docstring = False
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.rstrip("\n")
            stripped = line.strip()

            # Track docstring boundaries first so a closing line still
            # counts as docstring content.
            triple_count = stripped.count('"""')
            was_in_docstring = in_docstring
            if triple_count:
                # Toggle once per occurrence; an even count on a single
                # line leaves the flag unchanged (single-line docstring).
                if triple_count % 2 == 1:
                    in_docstring = not in_docstring

            if was_in_docstring or in_docstring:
                continue
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            yield lineno, line


def _scan(
    scopes: list[Path],
    pattern: re.Pattern[str],
    exclude: set[Path] | None = None,
) -> list[str]:
    """Return ``file:lineno: line`` hits for ``pattern`` across ``scopes``.

    ``exclude`` is a set of resolved file paths to skip entirely.
    """
    exclude = exclude or set()
    hits: list[str] = []
    seen: set[Path] = set()
    for scope in scopes:
        for path in _iter_py_files(scope):
            resolved = path.resolve()
            if resolved in exclude or resolved in seen:
                continue
            seen.add(resolved)
            for lineno, line in _code_lines(path):
                if pattern.search(line):
                    rel = path.relative_to(REPO_ROOT)
                    hits.append(f"{rel}:{lineno}: {line.strip()}")
    return hits


def _fmt(hits: list[str]) -> str:
    return "\n  " + "\n  ".join(hits) if hits else ""


# ----------------------------------------------------------------------
# Common scope sets
# ----------------------------------------------------------------------

ARCH_AND_FILL = [
    TOKENIZER / "arch",
    TOKENIZER / "fill_constant_candidates.py",
]

ARCH_ONLY = [TOKENIZER / "arch"]

GHIDRA_VIEWS = (TOKENIZER / "disasm" / "ghidra_views.py").resolve()
GHIDRA_PROVIDER = (TOKENIZER / "disasm" / "ghidra_provider.py").resolve()
ANGR_PROVIDER = (TOKENIZER / "disasm" / "angr_provider.py").resolve()


# ----------------------------------------------------------------------
# Guards
# ----------------------------------------------------------------------

def test_no_capstone_shape_reads():
    """``block.capstone.insns`` and ``insn.insn.insn_name`` are
    Capstone-internal shapes; arch consumers must read through the
    typed view (``block.instructions``, ``insn.base_mnemonic``)."""
    pattern = re.compile(r"block\.capstone\.insns|insn\.insn\.insn_name\b")
    hits = _scan(ARCH_AND_FILL, pattern)
    assert hits == [], f"Capstone-shape reads in consumer code:{_fmt(hits)}"


def test_no_op_type_magic_ints():
    """``op.type == <int>`` compares against Capstone enum integers
    directly; consumers must use the typed ``OpKind`` enum."""
    pattern = re.compile(r"op\.type == [0-9]+")
    hits = _scan(ARCH_AND_FILL, pattern)
    assert hits == [], f"Magic-int op.type compares in consumer code:{_fmt(hits)}"


def test_no_legacy_op_constants():
    """``_OP_REG = 1`` / ``_OP_IMM = 2`` / ``_OP_MEM = 3`` were
    Capstone-magic aliases the refactor removed in favour of
    ``OpKind``."""
    pattern = re.compile(r"_OP_REG\s*=\s*1|_OP_IMM\s*=\s*2|_OP_MEM\s*=\s*3")
    hits = _scan(ARCH_ONLY, pattern)
    assert hits == [], f"Legacy op-type integer aliases present:{_fmt(hits)}"


def test_no_cap_family_classes():
    """The ``_Cap*`` / ``_GhidraMemRawData`` / ``ghidra_raw_data`` /
    ``_LegacyCompat`` names belonged to the pre-refactor compat shim
    layer; nothing outside the ghidra view module itself should
    reference them."""
    pattern = re.compile(
        r"_CapFunction|_CapBlock|_CapInstruction|_CapOperand"
        r"|_CapShift|_CapMemOperand|_CapInsnInner"
        r"|_GhidraMemRawData|ghidra_raw_data|_LegacyCompat"
    )
    # Scope-fence per task spec: stale doc-comment historical
    # references in ghidra_views.py and ghidra_provider.py are
    # acceptable. _code_lines() already strips doc/comment lines, so
    # the exclusion is belt-and-suspenders for any code-level historic
    # mention we don't want to police here.
    hits = _scan(
        [TOKENIZER],
        pattern,
        exclude={GHIDRA_VIEWS, GHIDRA_PROVIDER},
    )
    assert hits == [], f"Legacy Cap-family / raw-data references:{_fmt(hits)}"


def test_no_meta_get_string_keys():
    """``meta.get('...')`` is the pre-refactor untyped dict access;
    the typed ``AddressMetadataView`` exposes named properties
    instead."""
    pattern = re.compile(r"meta\.get\(['\"]")
    scopes = [
        TOKENIZER / "constant_handler.py",
        TOKENIZER / "fill_constant_candidates.py",
    ]
    hits = _scan(scopes, pattern)
    assert hits == [], f"Stringly-keyed meta.get() in typed consumer:{_fmt(hits)}"


def test_no_format_version_inline_dispatch():
    """Inline ``format_version`` branching to choose v1 vs v2 dataclass
    types was confined to ``vm.ValuedConst`` / ``vm.BlockId``
    dispatchers; arch consumers must not re-introduce the conditional."""
    pattern = re.compile(
        r"format_version == 2.*Valued_Const"
        r"|Valued_Const_V2 if .*format_version"
        r"|Block_V2 if .*format_version"
    )
    hits = _scan(ARCH_AND_FILL, pattern)
    assert hits == [], f"Inline format_version dispatch in consumer code:{_fmt(hits)}"


def test_no_consumer_side_ghidra_raw_data():
    """``getattr(op, 'ghidra_raw_data')`` was a duck-typed escape hatch
    into provider internals; the typed operand view replaces it."""
    pattern = re.compile(
        r"getattr\(op, *\"ghidra_raw_data\""
        r"|getattr\(op, *'ghidra_raw_data'"
    )
    hits = _scan(ARCH_AND_FILL, pattern)
    assert hits == [], f"Consumer-side ghidra_raw_data getattr:{_fmt(hits)}"


def test_no_consumer_reg_name():
    """``insn.reg_name(...)`` is Capstone's reverse-lookup API and is
    not part of the cross-provider reusable view. Arch consumers must
    obtain register names through the typed operand view; only the
    provider module is allowed to call it directly."""
    pattern = re.compile(r"insn\.reg_name\(")
    hits = _scan(ARCH_ONLY, pattern)
    assert hits == [], f"Consumer-side insn.reg_name() call:{_fmt(hits)}"


def test_no_eager_list_materialization():
    """Hot-path lazy views must not be materialized with ``list(...)``;
    the wrappers are reused across iteration and copying them defeats
    the reuse contract. Internal use inside the providers themselves
    is fine (they own the underlying storage)."""
    pattern = re.compile(
        r"list\(.*\.operands\)"
        r"|list\(.*\.prefixes\)"
        r"|list\(.*\.instructions\)"
        r"|list\(.*\.blocks\)"
    )
    hits = _scan(
        [TOKENIZER],
        pattern,
        exclude={GHIDRA_VIEWS, ANGR_PROVIDER},
    )
    assert hits == [], f"Eager list(...) on hot-path view:{_fmt(hits)}"


def test_no_arch_x86_ghidra_operands():
    """``tokenizer/arch/x86/ghidra/operands.py`` was deleted in
    Phase G.3 — its responsibilities moved into the shared typed
    operand view."""
    stale = TOKENIZER / "arch" / "x86" / "ghidra" / "operands.py"
    assert not stale.exists(), (
        f"Stale Phase-G.3 file resurfaced: {stale.relative_to(REPO_ROOT)}"
    )


def test_no_operand_fp_width_bytes():
    """``operand_fp_width_bytes`` was replaced by ``operand_fp_type``
    in Phase F.1; no reference should survive anywhere in the
    package."""
    pattern = re.compile(r"operand_fp_width_bytes")
    hits = _scan([TOKENIZER], pattern)
    assert hits == [], f"Stale operand_fp_width_bytes reference:{_fmt(hits)}"


def test_no_reg_list_writeback_old_name():
    """``REG_LIST_WRITEBACK`` was renamed to ``asm_writeback_detect``
    in commit 628a33c — guard against accidental revert."""
    pattern = re.compile(r"REG_LIST_WRITEBACK")
    scope = TOKENIZER / "tokens.py"
    hits = _scan([scope], pattern)
    assert hits == [], f"Old REG_LIST_WRITEBACK token name in tokens.py:{_fmt(hits)}"
