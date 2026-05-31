"""Function cursor + its blocks container view.

Owns:
- ``_GhidraFunctionView``: reusable function wrapper.
- ``_GhidraBlocksView``: container view over a function's blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Iterator, Optional

from tokenizer.disasm.ghidra_views.block import _GhidraBlockView
from tokenizer.disasm.ghidra_views.unnamed_rename import placeholder_renamed_name
from tokenizer.disasm.types import Architecture, BlocksView
from tokenizer.function_deduper import (
    ThunkIdentity,
    ThunkTargetKind,
    canonical_function_name,
)


@dataclass(frozen=True, slots=True)
class FunctionIdentity:
    """The four identity axes of a function, derived ONCE by the provider.

    The provider's ``iter_functions`` must derive ``canonical_name`` up
    front for every function in order to sort by it (a thunk's canonical
    name is its real identity key, which the raw Ghidra name does not
    track — see :func:`tokenizer.function_deduper.canonical_function_name`).
    Since the three input axes (``name`` post-placeholder-rename,
    ``comment``, ``identity_key``) are already in hand at that point, all
    four are bundled here and threaded into the reused
    :class:`_GhidraFunctionView` via ``_advance``.

    This makes the sort key and the name ``main_loop`` writes to CSV
    column 0 structurally identical — there is a single ``canonical_name``
    string, not two independently-derived ones kept in sync by hope. Every
    consumer (the sort, the view, ``main_loop``) reads this one value.
    """

    name: str
    comment: Optional[str]
    identity_key: Optional[Hashable]
    canonical_name: str


# ---------------------------------------------------------------------------
# Container views (NOT lists - iterate, reuse child wrapper)
# ---------------------------------------------------------------------------
class _GhidraBlocksView:
    """Container view over a function's blocks.

    NOT a list. ``__iter__`` walks the function's Ghidra block iterator
    and yields the same reused ``_GhidraBlockView`` per block.
    """

    __slots__ = ("_function",)

    def __init__(self, function: "_GhidraFunctionView") -> None:
        self._function = function

    def __len__(self) -> int:
        return self._function._block_count

    def __iter__(self) -> Iterator["_GhidraBlockView"]:
        return self._function._iter_blocks()


# ---------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------
class _GhidraFunctionView:
    """Reusable function wrapper.

    Holds a single backing ``Ghidra Function`` reference. The block model
    is provider-supplied (constructed once per provider, reused).
    """

    __slots__ = (
        "_arch",
        "_program",
        "_listing",
        "_reg_map",
        "_decode",
        "_block_model",
        "_monitor",
        "_ghidra_function",
        "_entry",
        "_name",
        "_block_count",
        "_identity_key",
        "_comment",
        "_canonical_name",
        "_block_view",
        "_blocks_view",
    )

    def __init__(
        self,
        arch: Architecture,
        program: Any,
        listing: Any,
        reg_map: Any,
        decode: Any,
        block_model: Any,
        monitor: Any,
    ) -> None:
        self._arch = arch
        self._program = program
        self._listing = listing
        self._reg_map = reg_map
        self._decode = decode
        self._block_model = block_model
        self._monitor = monitor
        self._ghidra_function: Optional[Any] = None
        self._entry: int = 0
        self._name: str = ""
        self._block_count: int = 0
        self._identity_key: Optional[Hashable] = None
        self._comment: Optional[str] = None
        self._canonical_name: str = ""
        self._block_view = _GhidraBlockView(arch, program, listing, reg_map, decode, self)
        self._blocks_view = _GhidraBlocksView(self)

    def _advance(
        self, ghidra_function: Any, block_count: int, identity: FunctionIdentity
    ) -> None:
        """Repoint at the next Ghidra Function.

        ``block_count`` is precomputed by the provider's iter_functions
        loop (so ``len(blocks_view)`` is O(1)).

        ``identity`` carries the four identity axes the provider already
        derived to sort by canonical name (see :class:`FunctionIdentity`).
        The view is a pure carrier: it stores them verbatim and never
        recomputes. In particular the DEFAULT-source placeholder rename
        (``FUN_<hex>`` / ``thunk_FUN_<hex>`` → opaque ``unnamed @<hash>``
        label; see :mod:`tokenizer.disasm.ghidra_views.unnamed_rename`)
        and the canonical-name derivation both happen ONCE in the
        provider; ``identity.name`` is the already-renamed name and
        ``identity.canonical_name`` is the already-derived canonical.
        """
        self._ghidra_function = ghidra_function
        self._entry = int(ghidra_function.getEntryPoint().getOffset())
        self._block_count = block_count
        self._name = identity.name
        self._comment = identity.comment
        self._identity_key = identity.identity_key
        self._canonical_name = identity.canonical_name

    def _iter_blocks(self) -> Iterator[_GhidraBlockView]:
        if self._ghidra_function is None:
            return
        body = self._ghidra_function.getBody()
        block_iter = self._block_model.getCodeBlocksContaining(body, self._monitor)
        view = self._block_view
        while block_iter.hasNext():
            gblock = block_iter.next()
            # Precount instructions for O(1) __len__ on InstructionsView.
            insn_iter = self._listing.getInstructions(gblock, True)
            count = 0
            while insn_iter.hasNext():
                ghidra_insn = insn_iter.next()
                if not body.contains(ghidra_insn.getAddress()):
                    continue
                count += 1
            view._advance(gblock, count)
            yield view

    @property
    def entry(self) -> int:
        return self._entry

    @property
    def name(self) -> str:
        return self._name

    @property
    def blocks(self) -> BlocksView:
        return self._blocks_view

    @property
    def identity_key(self) -> Optional[Hashable]:
        return self._identity_key

    @property
    def comment(self) -> Optional[str]:
        return self._comment

    @property
    def canonical_name(self) -> str:
        return self._canonical_name

    def __deepcopy__(self, memo) -> "_GhidraFunctionView":
        clone = _GhidraFunctionView(
            self._arch,
            self._program,
            self._listing,
            self._reg_map,
            self._decode,
            self._block_model,
            self._monitor,
        )
        clone._ghidra_function = self._ghidra_function
        clone._entry = self._entry
        clone._name = self._name
        clone._block_count = self._block_count
        clone._identity_key = self._identity_key
        clone._comment = self._comment
        clone._canonical_name = self._canonical_name
        return clone


# ---------------------------------------------------------------------------
# Identity-key extraction
# ---------------------------------------------------------------------------
def _ghidra_identity_key(ghidra_function: Any) -> Optional[ThunkIdentity]:
    """Return a stable :class:`ThunkIdentity` when Ghidra recognises this
    function as a thunk, else ``None``.

    Implements the ``FunctionView.identity_key`` contract (see
    ``tokenizer/disasm/types.py``). Two cases:

    * External-target thunk (``thunked.isExternal()`` is True). The
      resolved Function lives in Ghidra's per-binary EXTERNAL block;
      its entry-point offset is a link-order-dependent PLACEHOLDER —
      same source symbol gets a different placeholder offset across
      binaries, so the offset is NOT cross-binary stable. The imported
      symbol name (``thunked.getName()``) IS cross-binary stable for
      the same source symbol; that is the identity key.
    * Local-target thunk (``isExternal()`` is False — rare; hand-written
      assembly aliases, IFUNCs, some toolchain trampolines). When the
      target carries a real (non-DEFAULT) symbol source — the common
      sub-case — its ``getName()`` is cross-binary stable (Ghidra
      assigns the same name to the same source function across
      binaries) and we key on that. When the target itself is a
      DEFAULT-source placeholder (``FUN_xxx``-style) — the only sub-
      case where no cross-binary name exists — we fall back to the
      entry-point offset, which is within-binary stable. The kind axis
      stays LOCAL in both sub-cases so cross-kind collisions remain
      impossible.

    Non-thunk functions return ``None`` (legacy disambiguation path —
    the provider declines to assert identity beyond name).

    Resilient to partially-populated Ghidra programs: any exception
    from the Java side is swallowed and the function falls back to
    ``None`` (= "no merge"), preserving the legacy behaviour rather
    than crashing the iter loop.
    """
    is_thunk = getattr(ghidra_function, "isThunk", None)
    if is_thunk is None or not bool(is_thunk()):
        return None
    get_thunked = getattr(ghidra_function, "getThunkedFunction", None)
    if get_thunked is None:
        return None
    try:
        thunked = get_thunked(True)
    except Exception:
        return None
    if thunked is None:
        return None
    try:
        is_external = bool(thunked.isExternal())
    except Exception:
        return None
    if is_external:
        # Imported-symbol name is cross-binary stable.
        try:
            name = str(thunked.getName())
        except Exception:
            return None
        return ThunkIdentity(kind=ThunkTargetKind.EXTERNAL, key=name)
    # Local-target thunk: prefer the target function's name (cross-
    # binary stable when the target carries a real symbol source —
    # USER_DEFINED / IMPORTED / ANALYSIS), fall back to the entry-
    # point offset only when the target itself is DEFAULT-source
    # (``FUN_xxx``-style placeholder). String-compare on
    # ``str(getSource())`` mirrors the ``_DEFAULT_SOURCE_STR`` pattern
    # in ``unnamed_rename.py`` and shields the call site from JPype's
    # evolving enum-import idioms.
    try:
        target_symbol = thunked.getSymbol()
        target_source = target_symbol.getSource() if target_symbol is not None else None
    except Exception:
        target_source = None
    if target_source is not None and str(target_source) != "DEFAULT":
        try:
            name = str(thunked.getName())
        except Exception:
            return None
        return ThunkIdentity(kind=ThunkTargetKind.LOCAL, key=name)
    try:
        offset = int(thunked.getEntryPoint().getOffset())
    except Exception:
        return None
    return ThunkIdentity(kind=ThunkTargetKind.LOCAL, key=f"{offset:x}")


# ---------------------------------------------------------------------------
# Comment (plate-comment) extraction
# ---------------------------------------------------------------------------
def _ghidra_function_comment(ghidra_function: Any) -> Optional[str]:
    """Return the function's plate comment (when set) or ``None``.

    Implements the ``FunctionView.comment`` contract (see
    ``tokenizer/disasm/types.py``): for C++ symbols Ghidra's demangler
    populates the plate comment with the demangled scoped signature
    (e.g. ``ARPHeader::storeRecvData(unsigned char const*, unsigned int)``).
    That string is the natural context disambiguator when two distinct
    methods share an unqualified name (``ARPHeader::reset`` vs
    ``EthernetHeader::reset``). Returns ``None`` when no plate comment
    is set (the common case for C / asm symbols and for any function
    whose demangler entry the analysis did not run).

    Resilient to partially-populated Ghidra programs: any exception
    from the Java side is swallowed and the function falls back to
    ``None``, preserving the legacy behaviour rather than crashing
    the iter loop (matches the defensive pattern in
    :func:`_ghidra_identity_key`).
    """
    get_comment = getattr(ghidra_function, "getComment", None)
    if get_comment is None:
        return None
    try:
        raw = get_comment()
    except Exception:
        return None
    if raw is None:
        return None
    try:
        return str(raw)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Identity derivation (provider per-function step)
# ---------------------------------------------------------------------------
def _derive_function_identity(
    ghidra_function: Any, binary_id_hash: bytes
) -> FunctionIdentity:
    """Derive the four identity axes for one Ghidra function.

    This is the provider's per-function identity step, factored out of
    ``iter_functions`` so the placeholder-rename + canonical-name wiring
    is unit-testable without a live Ghidra program. It composes the three
    extractors in this module — the DEFAULT-source placeholder rename
    (:func:`...unnamed_rename.placeholder_renamed_name`),
    :func:`_ghidra_function_comment`, :func:`_ghidra_identity_key` — and
    renders the cross-ISA-stable canonical name from them via
    :func:`tokenizer.function_deduper.canonical_function_name`.

    The result is threaded into the reused :class:`_GhidraFunctionView`
    (so the view never recomputes) and is the SINGLE derivation of the
    canonical name the provider sorts by and ``main_loop`` writes.
    """
    raw_name = str(ghidra_function.getName())
    source = ghidra_function.getSymbol().getSource()
    name = placeholder_renamed_name(raw_name, source, binary_id_hash)
    comment = _ghidra_function_comment(ghidra_function)
    identity_key = _ghidra_identity_key(ghidra_function)
    return FunctionIdentity(
        name=name,
        comment=comment,
        identity_key=identity_key,
        canonical_name=canonical_function_name(name, comment, identity_key),
    )
