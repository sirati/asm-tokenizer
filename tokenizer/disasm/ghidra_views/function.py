"""Function cursor + its blocks container view.

Owns:
- ``_GhidraFunctionView``: reusable function wrapper.
- ``_GhidraBlocksView``: container view over a function's blocks.
"""

from __future__ import annotations

from typing import Any, Hashable, Iterator, Optional

from tokenizer.disasm.ghidra_views.block import _GhidraBlockView
from tokenizer.disasm.types import Architecture, BlocksView


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
        "_identity_key_resolved",
        "_comment",
        "_comment_resolved",
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
        self._identity_key_resolved: bool = False
        self._comment: Optional[str] = None
        self._comment_resolved: bool = False
        self._block_view = _GhidraBlockView(arch, program, listing, reg_map, decode, self)
        self._blocks_view = _GhidraBlocksView(self)

    def _advance(self, ghidra_function: Any, block_count: int) -> None:
        """Repoint at the next Ghidra Function.

        ``block_count`` is precomputed by the provider's iter_functions
        loop (so ``len(blocks_view)`` is O(1)).
        """
        self._ghidra_function = ghidra_function
        self._entry = int(ghidra_function.getEntryPoint().getOffset())
        self._name = str(ghidra_function.getName())
        self._block_count = block_count
        # Lazy fields: compute on first property access. Reset the cache
        # flags so the next access reruns _ghidra_identity_key /
        # _ghidra_function_comment against the new backing function.
        self._identity_key = None
        self._identity_key_resolved = False
        self._comment = None
        self._comment_resolved = False

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
        if not self._identity_key_resolved:
            self._identity_key = _ghidra_identity_key(self._ghidra_function)
            self._identity_key_resolved = True
        return self._identity_key

    @property
    def comment(self) -> Optional[str]:
        if not self._comment_resolved:
            self._comment = _ghidra_function_comment(self._ghidra_function)
            self._comment_resolved = True
        return self._comment

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
        clone._identity_key_resolved = self._identity_key_resolved
        clone._comment = self._comment
        clone._comment_resolved = self._comment_resolved
        return clone


# ---------------------------------------------------------------------------
# Identity-key extraction
# ---------------------------------------------------------------------------
def _ghidra_identity_key(ghidra_function: Any) -> Optional[Hashable]:
    """Return a stable identity key when Ghidra recognises this function
    as a thunk, else ``None``.

    Implements the ``FunctionView.identity_key`` contract (see
    ``tokenizer/disasm/types.py``): for PLT-thunk functions the key is
    the resolved external's entry-point offset
    (``Function.getThunkedFunction(True).getEntryPoint().getOffset()``),
    which is identical across every trampoline slot that resolves to
    the same external symbol AND stable across ISA variants. For
    non-thunk functions the provider declines to assert identity beyond
    name and returns ``None`` (legacy disambiguation path).

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
        return int(thunked.getEntryPoint().getOffset())
    except Exception:
        return None


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
