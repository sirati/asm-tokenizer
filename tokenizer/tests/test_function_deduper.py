"""Tests for the per-binary semantic-merge gate.

Two concerns under test, separated cleanly:

1. ``FunctionDeduper`` (in :mod:`tokenizer.function_deduper`) - the
   gate itself. The four-way merge condition (same name + same comment
   + same identity_key + same content) and the slot_id allocation
   contract.

2. ``FunctionDataManager.add_function_data`` (in
   :mod:`tokenizer.function_data_manager`) integration with the gate.
   Same-key duplicate folds into the existing FID and returns its
   final name; differing comment / identity_key / body keeps the
   legacy ``_N``-suffix path.
"""

from __future__ import annotations

from tokenizer.function_data_manager import FunctionData, FunctionDataManager
from tokenizer.function_deduper import (
    DedupResolution,
    FunctionDeduper,
    canonical_function_name,
    logical_function_name,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fd(tokens_base64: str) -> FunctionData:
    """Build a minimal FunctionData carrying a controllable token body.

    The deduper only consults ``tokens_base64``; the other fields are
    inert for these tests so we pass ``None`` placeholders (the
    dataclass runtime is structural, not type-checked).
    """
    return FunctionData(
        tokens=None,  # type: ignore[arg-type]
        tokens_base64=tokens_base64,
        block_runlength_base64="",
        instruction_runlength_base64="",
        metadata_cell="",
    )


# ---------------------------------------------------------------------------
# FunctionDeduper - the gate
# ---------------------------------------------------------------------------


def test_first_call_returns_distinct_with_slot_id() -> None:
    """A first encounter always returns ``is_duplicate=False`` with a
    freshly-allocated ``slot_id``."""
    deduper = FunctionDeduper()
    r = deduper.resolve("strcmp", None, 0xDEAD, "AAAA")
    assert isinstance(r, DedupResolution)
    assert r.is_duplicate is False
    assert r.body_divergence_warning is False
    assert r.slot_id == 0


def test_four_way_match_folds_with_same_slot_id() -> None:
    """Same name + same comment + same identity_key + same body =
    MERGE; ``is_duplicate=True`` and the second call's ``slot_id``
    matches the first."""
    deduper = FunctionDeduper()
    r1 = deduper.resolve("strcmp", None, 0xDEAD, "AAAA")
    r2 = deduper.resolve("strcmp", None, 0xDEAD, "AAAA")
    assert r2.is_duplicate is True
    assert r2.slot_id == r1.slot_id


def test_different_comment_disambiguates() -> None:
    """Same name + different comment ⇒ DISTINCT records (the canonical
    C++-overload case)."""
    deduper = FunctionDeduper()
    r1 = deduper.resolve(
        "reset", "ARPHeader::reset(void)", None, "AAAA"
    )
    r2 = deduper.resolve(
        "reset", "EthernetHeader::reset(void)", None, "BBBB"
    )
    assert r1.is_duplicate is False
    assert r2.is_duplicate is False
    assert r1.slot_id != r2.slot_id


def test_same_comment_merges_when_body_matches() -> None:
    """Same name + same comment + same body ⇒ MERGE (overload's PLT
    trampoline collapse — Ghidra's analysis can produce multiple
    Function handles with identical demangled signature when the
    demangler+analysis tag the same logical method twice)."""
    deduper = FunctionDeduper()
    r1 = deduper.resolve(
        "reset", "ARPHeader::reset(void)", None, "AAAA"
    )
    r2 = deduper.resolve(
        "reset", "ARPHeader::reset(void)", None, "AAAA"
    )
    assert r1.is_duplicate is False
    assert r2.is_duplicate is True
    assert r1.slot_id == r2.slot_id


def test_same_name_different_identity_key_distinct_records() -> None:
    """A second call with the same name but a different identity_key
    is a genuine collision (different external resolved-to); two
    distinct slot_ids."""
    deduper = FunctionDeduper()
    r1 = deduper.resolve("strcmp", None, 0xDEAD, "AAAA")
    r2 = deduper.resolve("strcmp", None, 0xBEEF, "AAAA")
    assert r1.is_duplicate is False
    assert r2.is_duplicate is False
    assert r1.slot_id != r2.slot_id


def test_body_divergence_warning_path() -> None:
    """Same name + same comment + same identity_key but DIFFERENT body
    is the warning case: the first body's slot stays canonical;
    subsequent divergent bodies surface
    ``body_divergence_warning=True`` and spawn a fresh slot_id."""
    deduper = FunctionDeduper()
    r1 = deduper.resolve("strcmp", None, 0xDEAD, "AAAA")
    r2 = deduper.resolve("strcmp", None, 0xDEAD, "BBBB")
    assert r1.is_duplicate is False
    assert r2.is_duplicate is False
    assert r2.body_divergence_warning is True
    assert r2.slot_id != r1.slot_id


def test_first_recorded_body_wins_after_divergence() -> None:
    """When a same-identity-tuple call surfaces with a divergent body
    (rejected fold), the gate keeps the FIRST body as the canonical
    one. A subsequent matching call against the first body still
    folds; a matching call against the second does not."""
    deduper = FunctionDeduper()
    r1 = deduper.resolve("strcmp", None, 0xDEAD, "AAAA")
    deduper.resolve("strcmp", None, 0xDEAD, "BBBB")
    r3 = deduper.resolve("strcmp", None, 0xDEAD, "AAAA")
    assert r3.is_duplicate is True
    assert r3.slot_id == r1.slot_id


def test_lto_clone_no_comment_no_key_same_body_merges() -> None:
    """The LTO-clone case: comment=None + identity_key=None + same
    body ⇒ MERGE under the same slot_id (body-equivalence merge for
    LTO-emitted static-helper clones inlined into many TUs)."""
    deduper = FunctionDeduper()
    r1 = deduper.resolve("gh_lnode_next", None, None, "AAAA")
    r2 = deduper.resolve("gh_lnode_next", None, None, "AAAA")
    assert r1.is_duplicate is False
    assert r2.is_duplicate is True
    assert r1.slot_id == r2.slot_id


def test_lto_clone_no_comment_no_key_different_body_warns() -> None:
    """The unsafe residual case: comment=None + identity_key=None +
    different body ⇒ DISTINCT slot_ids via the body-divergence-warning
    path (rare cross-binary inconsistency the caller may log)."""
    deduper = FunctionDeduper()
    r1 = deduper.resolve("gh_lnode_next", None, None, "AAAA")
    r2 = deduper.resolve("gh_lnode_next", None, None, "BBBB")
    assert r1.is_duplicate is False
    assert r2.is_duplicate is False
    assert r2.body_divergence_warning is True
    assert r1.slot_id != r2.slot_id


def test_empty_comment_normalises_to_none() -> None:
    """Some Ghidra Function handles surface an empty string for
    ``getComment()``; the deduper treats it as ``None`` so the C++
    merge path doesn't fire on noise."""
    deduper = FunctionDeduper()
    r1 = deduper.resolve("foo", "", None, "AAAA")
    r2 = deduper.resolve("foo", None, None, "AAAA")
    # Same (name, treated-comment, identity_key, body) ⇒ MERGE.
    assert r2.is_duplicate is True
    assert r2.slot_id == r1.slot_id


# ---------------------------------------------------------------------------
# FunctionDataManager - the gate integrated into the manager API
# ---------------------------------------------------------------------------


def test_fdm_same_name_same_key_same_body_folds() -> None:
    """The merge condition holds: the second call is folded (no new
    slot consumed, the first record's final name is returned). With
    ``identity_key=0xDEAD`` the canonical name is the thunk-keyed
    form; both calls return the same canonical."""
    mgr = FunctionDataManager(total_functions=4)
    final_a = mgr.add_function_data(
        "strcmp", 0x1000, "disas_a", "tok_a", _fd("AAAA"), identity_key=0xDEAD
    )
    final_b = mgr.add_function_data(
        "strcmp", 0x2000, "disas_b", "tok_b", _fd("AAAA"), identity_key=0xDEAD
    )
    canonical = canonical_function_name("strcmp", None, 0xDEAD)
    assert final_a == canonical
    assert final_b == canonical
    assert mgr.get_used_count() == 1


def test_fdm_different_comment_keeps_distinct_records() -> None:
    """C++ overload disambiguation: same name + different comment ⇒
    two distinct FDM records under DISTINCT canonical names (the
    canonical-name helper produces the cross-ISA-stable suffix from
    the demangled comment, so two genuine overloads never collide on
    the on-disk name)."""
    mgr = FunctionDataManager(total_functions=4)
    final_a = mgr.add_function_data(
        "reset", 0x1000, "disas_a", "tok_a", _fd("AAAA"),
        comment="ARPHeader::reset(void)",
    )
    final_b = mgr.add_function_data(
        "reset", 0x2000, "disas_b", "tok_b", _fd("BBBB"),
        comment="EthernetHeader::reset(void)",
    )
    assert final_a == canonical_function_name(
        "reset", "ARPHeader::reset(void)", None
    )
    assert final_b == canonical_function_name(
        "reset", "EthernetHeader::reset(void)", None
    )
    assert final_a != final_b
    assert mgr.get_used_count() == 2


def test_fdm_same_name_different_identity_key_distinct() -> None:
    """Different identity_key (genuine collision) preserves two
    distinct records under DISTINCT thunk-keyed canonical names."""
    mgr = FunctionDataManager(total_functions=4)
    final_a = mgr.add_function_data(
        "strcmp", 0x1000, "disas_a", "tok_a", _fd("AAAA"), identity_key=0xDEAD
    )
    final_b = mgr.add_function_data(
        "strcmp", 0x2000, "disas_b", "tok_b", _fd("AAAA"), identity_key=0xBEEF
    )
    assert final_a == canonical_function_name("strcmp", None, 0xDEAD)
    assert final_b == canonical_function_name("strcmp", None, 0xBEEF)
    assert final_a != final_b
    assert mgr.get_used_count() == 2


def test_fdm_body_divergence_keeps_distinct_records() -> None:
    """Same (name, comment, identity_key) but different body: NOT
    folded (regression-guards the body-equality condition); the
    second slot consumes a ``_N``-suffix fallback on the SHARED
    canonical name (body-divergence diagnostic — same canonical name
    two distinct bodies)."""
    mgr = FunctionDataManager(total_functions=4)
    final_a = mgr.add_function_data(
        "strcmp", 0x1000, "disas_a", "tok_a", _fd("AAAA"), identity_key=0xDEAD
    )
    final_b = mgr.add_function_data(
        "strcmp", 0x2000, "disas_b", "tok_b", _fd("BBBB"), identity_key=0xDEAD
    )
    canonical = canonical_function_name("strcmp", None, 0xDEAD)
    assert final_a == canonical
    assert final_b == f"{canonical}_1"
    assert mgr.get_used_count() == 2


def test_fdm_lto_clone_no_key_no_comment_same_body_folds() -> None:
    """LTO-clone merge via body-equivalence: same name + None comment
    + None identity_key + same body ⇒ FOLD (no slot consumed). This is
    the change Phase B-2 introduces — pre-patch the gate short-circuited
    to "no merge" on ``identity_key=None``."""
    mgr = FunctionDataManager(total_functions=4)
    final_a = mgr.add_function_data(
        "gh_lnode_next", 0x1000, "disas_a", "tok_a", _fd("AAAA")
    )
    final_b = mgr.add_function_data(
        "gh_lnode_next", 0x2000, "disas_b", "tok_b", _fd("AAAA")
    )
    assert final_a == "gh_lnode_next"
    assert final_b == "gh_lnode_next"
    assert mgr.get_used_count() == 1


def test_fdm_none_identity_key_different_body_distinct() -> None:
    """``identity_key=None`` + ``comment=None`` + different body: NOT
    folded (legacy ``_N``-suffix path applies)."""
    mgr = FunctionDataManager(total_functions=4)
    final_a = mgr.add_function_data(
        "ctor", 0x1000, "disas_a", "tok_a", _fd("AAAA")
    )
    final_b = mgr.add_function_data(
        "ctor", 0x2000, "disas_b", "tok_b", _fd("BBBB")
    )
    assert final_a == "ctor"
    assert final_b == "ctor_1"
    assert mgr.get_used_count() == 2


def test_fdm_fold_returns_existing_address_through_lookup() -> None:
    """A folded second call leaves the first record's address as the
    canonical mapping for that name (the second slot's address is
    intentionally dropped; both call sites that reference either
    address resolve to the same logical function). The lookup key is
    the CANONICAL name (the on-disk form), not the raw provider
    name."""
    mgr = FunctionDataManager(total_functions=4)
    mgr.add_function_data(
        "strcmp", 0x1000, "disas_a", "tok_a", _fd("AAAA"), identity_key=0xDEAD
    )
    mgr.add_function_data(
        "strcmp", 0x2000, "disas_b", "tok_b", _fd("AAAA"), identity_key=0xDEAD
    )
    canonical = canonical_function_name("strcmp", None, 0xDEAD)
    assert mgr.get_function_addr(canonical, 0) == 0x1000


# ---------------------------------------------------------------------------
# canonical_function_name - the cross-ISA-stable name derivation helper
# ---------------------------------------------------------------------------


def test_canonical_name_no_comment_no_identity_returns_raw() -> None:
    """The both-axes-None branch returns the raw name unchanged so
    legacy callers (and the FDM positional-fallback path) keep the
    identical semantics."""
    assert canonical_function_name("foo", None, None) == "foo"


def test_canonical_name_empty_comment_treated_as_none() -> None:
    """An empty-string comment behaves identically to ``None`` so
    providers that surface ``getComment() == ""`` for noise don't
    accidentally produce ``foo@``-style suffixes."""
    assert canonical_function_name("foo", "", None) == "foo"


def test_canonical_name_comment_populated_appends_sanitised_suffix() -> None:
    """A populated comment becomes the cross-ISA-stable suffix; the
    output is human-readable (allow-listed chars survive verbatim)."""
    result = canonical_function_name(
        "reset", "ARPHeader::reset(void)", None
    )
    assert result == "reset@ARPHeader::reset(void)"


def test_canonical_name_comment_with_whitespace_collapses() -> None:
    """Whitespace runs in the comment collapse to a single ``_`` so
    the suffix stays one record (no CSV-cell-corrupting characters).
    Adjacent unsafe-char runs (e.g. comma + space) also collapse to a
    single ``_`` rather than producing decorative ``__`` clusters."""
    result = canonical_function_name(
        "store",
        "ARPHeader::storeRecvData(unsigned char const*, unsigned int)",
        None,
    )
    # spaces between argument-list tokens collapse; the structural
    # ``::``, ``(``, ``)``, ``*`` survive. The ``,`` + space pair
    # collapses to a single ``_`` (both chars are non-allow-listed).
    assert result == (
        "store@ARPHeader::storeRecvData(unsigned_char_const*_unsigned_int)"
    )


def test_canonical_name_comment_with_unsafe_chars_replaced() -> None:
    """Characters outside the allow-list (commas, slashes, quotes,
    newlines) get replaced with ``_`` so the suffix is safe in CSV
    cells, sidecar lines, and filesystem paths."""
    result = canonical_function_name("f", 'name "quoted",\nx/y', None)
    # ``"`` and ``,`` and ``\n`` and ``/`` -> ``_``; whitespace runs
    # collapse; leading/trailing ``_`` stripped.
    assert result == "f@name_quoted_x_y"


def test_canonical_name_thunk_identity_key_branch() -> None:
    """The comment=None + identity_key=populated branch produces the
    thunk-keyed suffix (cross-ISA-stable: the resolved-external entry
    offset is identical across thunks AND across ISA variants)."""
    assert canonical_function_name("strcmp", None, 0xDEAD) == "strcmp@thunk:57005"


def test_canonical_name_comment_takes_precedence_over_identity_key() -> None:
    """When both axes are populated the comment wins (the demangled
    signature is the strictly stronger disambiguator; identity_key
    only helps when the demangler is silent)."""
    result = canonical_function_name(
        "f", "C::m()", 0xDEAD
    )
    assert result == "f@C::m()"


def test_canonical_name_long_comment_truncated_with_hash_suffix() -> None:
    """A pathologically long comment gets truncated with a ``~<sha1>``
    suffix so the canonical name fits in CSV cells / filesystem path
    components. The truncation is deterministic (same input -> same
    suffix) so the cross-ISA-stable property survives the cap."""
    long_comment = "A::" + "x" * 500
    r1 = canonical_function_name("f", long_comment, None)
    r2 = canonical_function_name("f", long_comment, None)
    assert r1 == r2  # deterministic
    assert "~" in r1
    # 200-char cap: prefix (192) + ``~`` + 7-char hex digest = 200; +
    # ``f@`` prefix = 202.
    assert len(r1) <= 1 + len("f@") + 200


def test_canonical_name_is_deterministic() -> None:
    """Same inputs -> same output, always (cross-ISA-stable requires
    this)."""
    a = canonical_function_name("foo", "C::m()", 0xDEAD)
    b = canonical_function_name("foo", "C::m()", 0xDEAD)
    assert a == b


# ---------------------------------------------------------------------------
# ``logical_function_name`` -- the inverse used by the inspector function
# list to collapse PLT-thunk variants whose resolved-extern offset
# differs across ELF builds.
# ---------------------------------------------------------------------------


def test_logical_name_strips_thunk_suffix() -> None:
    """``@thunk:<digits>`` is the only suffix logical-name peels off; it
    is the one canonical suffix that is NOT cross-ISA-stable in practice
    (Ghidra's resolved-extern entry offset varies per binary)."""
    assert logical_function_name("gzseek@thunk:1056296") == "gzseek"
    assert logical_function_name("adler32_combine@thunk:1105632") == "adler32_combine"


def test_logical_name_preserves_plain_name() -> None:
    """Names without the suffix pass through verbatim; no fast-path
    branching needed at the call site."""
    assert logical_function_name("gzseek") == "gzseek"
    assert logical_function_name("main") == "main"


def test_logical_name_preserves_comment_suffix() -> None:
    """The ``@<sanitised_signature>`` suffix from
    :func:`canonical_function_name` is the demangled C++ signature and
    IS deterministic across builds; logical-name must keep it so two
    methods sharing an unqualified name stay distinct."""
    canon = canonical_function_name(
        "reset", "ARPHeader::reset(NetworkLayerElement*)", None
    )
    assert "@" in canon
    assert logical_function_name(canon) == canon


def test_logical_name_thunk_pairs_collapse_across_offsets() -> None:
    """The corpus symptom: multiple binaries report the same source
    symbol with different ``@thunk:<offset>`` because the resolved
    extern lives at different placeholder addresses per ELF. Logical
    name must collapse them to the underlying symbol."""
    a = canonical_function_name("gzseek", None, 1056296)
    b = canonical_function_name("gzseek", None, 1056324)
    assert a != b
    assert logical_function_name(a) == logical_function_name(b) == "gzseek"
