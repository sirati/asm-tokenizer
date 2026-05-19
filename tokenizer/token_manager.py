import typing
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import numpy.typing as npt

from tokenizer.architecture import PlatformInstructionTypes
from tokenizer.token_utils import TokenUtils
from tokenizer.tokens import (
    BFloat16Token,
    BlockDefToken,
    BlockToken,
    BlockTokenV2,
    CodePtrTableToken,
    ExtFuncToken,
    Float16Token,
    Float32Token,
    Float64Token,
    Float80Token,
    Float128Token,
    FloatToken,
    IdentifierToken,
    JumpTableToken,
    LitTokenType,
    LocalFuncToken,
    LocalFunctionToken,
    MemoryOperandSymbol,
    MemoryOperandToken,
    ModifierToken,
    OpaqueConstToken,
    PlatformToken,
    PltFuncToken,
    RegisterListSymbol,
    RegisterListToken,
    RoDataPtrToken,
    RwDataPtrToken,
    StringPtrToken,
    ThreadLocalToken,
    Tokens,
    TokenType,
    ValuedConstToken,
    ValuedConstTokenV2,
    VariantAxisToken,
    VtableToken,
)


# v2 inline-digit byte-packing helpers. These are the local fallback for
# the Phase 1.A.3 `TokenUtils.int_to_minimum_bytes` /
# `TokenUtils.decode_v2_inline_digits` helpers. Both 1.A.2 (this task)
# and 1.A.3 land independently on parallel branches and merge into the
# v2 base; once 1.A.3 is in, the orchestrator can switch the Inner
# classes below to call `TokenUtils.*` directly and drop these locals.
# Behavior contract:
#   _v2_int_to_minimum_bytes(0)   -> b'\x00'         (1 byte; never 0-byte)
#   _v2_int_to_minimum_bytes(255) -> b'\xff'
#   _v2_int_to_minimum_bytes(256) -> b'\x01\x00'
# Negative values are rejected at the caller; this helper is unsigned.
def _v2_int_to_minimum_bytes(value: int) -> bytes:
    """Encode a non-negative integer as minimum-width big-endian bytes (>=1 byte)."""
    if value < 0:
        raise ValueError(f"_v2_int_to_minimum_bytes is unsigned-only, got {value}")
    if value == 0:
        return b"\x00"
    width = (value.bit_length() + 7) // 8
    return value.to_bytes(width, "big")


def _v2_bytes_to_int(byte_ids: typing.Iterable[int]) -> int:
    """Decode a big-endian byte sequence (each int in 0..255) to an int."""
    result = 0
    for b in byte_ids:
        if not (0 <= b < 256):
            raise ValueError(f"v2 digit byte out of range: {b}")
        result = (result << 8) | b
    return result


class VocabularyManager:
    """Manages vocabulary for token-to-ID mapping"""

    # IDs 0..255 are protocol-reserved digit slots under format_version
    # in (1, 2) (v1 unified vocab + v2 per-binary CSV; both share the
    # inline-digit wire encoding). `_V2_RESERVED_DIGIT_COUNT` is the
    # literal boundary — the first vocab entry registered on such a VM
    # lands at id _V2_RESERVED_DIGIT_COUNT.
    _V2_RESERVED_DIGIT_COUNT = 256

    def __init__(self, platform: typing.Optional[str], _init=True, format_version: int = 1):
        self.platform = platform
        # Wire-format version: 1 = unified vocab (inline-digit stream;
        # see plan memoized-booping-wren.md — the new unified-vocab
        # format that supersedes the deleted legacy unified numbering);
        # 2 = per-binary CSV inline-digit category-token stream (see
        # plan vivid-tinkering-wilkes.md). Both v1 and v2 use the
        # inline-digit wire encoding, so `_private_add_token` skips IDs
        # 0..255 under either to keep them free for digit continuations
        # in the token stream. Per-binary CSV saver emits a
        # `format_version=2` prelude in vocab.csv. Any other
        # format_version value (e.g. the historical "1 = legacy
        # Block_<HEX> / Lit_Start framing" pathway) bypasses both the
        # reserved-digit pre-population and the V2 Inner-class
        # dispatch — used today only by ad-hoc callers that supply
        # `vocab_list` explicitly through `from_vocab`.
        self.format_version = format_version
        if platform is None:
            self.platform_list: list[str] = []
            self.platform_reverse: dict[str, int] = {}
            self.token_to_platform: npt.NDArray[np.int8] = np.full(256, -1, dtype=np.int8)

        self.token_to_id: dict[str, int] = {}  # dict: tokenstr to id
        if _init:
            self.id_to_token: list[str] = []  # array: id to tokenstr
            self.registry_token_cache: list[Tokens] = []  # registry cache

            # Preallocated numpy arrays with different initial capacities
            self._id_to_token_type: npt.NDArray[np.int8] = np.full(256, TokenType.ERROR, dtype=np.int8)

            # Smaller initial capacity for lit caches since they're sparse
            self._lit_start_cache: npt.NDArray[np.int_] = np.empty(4, dtype=np.int_)
            self._lit_end_cache: npt.NDArray[np.int_] = np.empty(4, dtype=np.int_)
            self._lit_start_count = 0  # Track actual entries in lit_start_cache
            self._lit_end_count = 0  # Track actual entries in lit_end_cache

            # New cache for platform instruction types
            self._platform_instruction_type_cache: npt.NDArray[np.int8] = np.full(
                256, PlatformInstructionTypes.AGNOSTIC, dtype=np.int8
            )

            # Under v2 (per-binary CSV) and v1 (the new unified vocab;
            # see plan memoized-booping-wren.md — v1 is the renumbered
            # unified vocab that takes over from the deleted legacy
            # unified format), pre-populate ids 0..255 with
            # debug-friendly `digit_<HH>` placeholders and mark their
            # token_type as UNRESOLVED so the array stays
            # self-consistent. The `token_to_id` dict deliberately
            # stays empty for these positions — they are addressed by
            # their literal numeric value in the token stream, not by
            # name. v1 unified reuses the same reserved-digit layout:
            # variant tokens land at IDs `_V2_RESERVED_DIGIT_COUNT`
            # (256) and up, identical to how v2 places its first real
            # entry.
            if format_version in (1, 2):
                self.id_to_token.extend(
                    f"digit_{i:02X}" for i in range(self._V2_RESERVED_DIGIT_COUNT)
                )
                self._id_to_token_type[: self._V2_RESERVED_DIGIT_COUNT] = TokenType.UNRESOLVED

        # Create unique inner classes for this instance
        self._create_inner_classes()

    @staticmethod
    def from_vocab(
        platform: str,
        vocab_list: list[str],
        platform_instruction_type_cache: npt.NDArray[np.int8],
        id_to_token_type: npt.NDArray[np.int8] = None,
        lit_start_cache: npt.NDArray[np.int_] = None,
        lit_end_cache: npt.NDArray[np.int_] = None,
        platform_list: list[str] = None,
        token_to_platform: npt.NDArray[np.int8] = None,
        format_version: int = 1,
    ) -> "VocabularyManager":
        """Creates vocab from tokenizer output.

        Callers passing `format_version` in (1, 2) (the inline-digit
        wire encoding — v1 unified vocab + v2 per-binary CSV) are
        responsible for supplying a `vocab_list` and `id_to_token_type`
        whose first 256 entries are the reserved digit slots (loaders
        reconstitute those from the protocol convention since vocab.csv
        writes no entries for them).
        """
        v_man = VocabularyManager(platform, format_version=format_version)
        # Reassigning id_to_token wholesale replaces the constructor's
        # placeholder population (intentional — the caller has the
        # authoritative list).
        v_man.id_to_token = vocab_list
        v_man.last_id = len(vocab_list)
        platform_token = f"{platform}_"

        v_man._platform_instruction_type_cache = platform_instruction_type_cache

        if platform_list is not None:
            assert platform is None
            v_man.platform_list = platform_list
            v_man.token_to_platform = token_to_platform
            for i, platform_value in enumerate(platform_list):
                v_man.platform_reverse[platform_value] = i

        assert ((id_to_token_type is None) == (lit_start_cache is None)) and (
            (lit_start_cache is None) == (lit_end_cache is None)
        ), "All or none of id_to_token_type, lit_start_cache, and lit_end_cache must be provided"

        if id_to_token_type is not None:
            v_man._id_to_token_type = id_to_token_type
            v_man._lit_start_cache = lit_start_cache
            v_man._lit_start_count = len(lit_start_cache)
            v_man._lit_end_cache = lit_end_cache
            v_man._lit_end_count = len(lit_end_cache)

            for index, value in enumerate(vocab_list):
                v_man.token_to_id[value] = index
        else:
            assert platform_list is None, "Not implemented yet"
            # Initialize numpy arrays with proper size
            token_types = []
            lit_start_tokens = []
            lit_end_tokens = []

            for index, value in enumerate(vocab_list):
                v_man.token_to_id[value] = index

                token_type: int = TokenType.ERROR
                if value.startswith(platform_token):
                    token_type = TokenType.PLATFORM
                elif value.startswith("VALUED_"):
                    token_type = TokenType.VALUED_CONST
                elif value == "Block_Def":
                    token_type = TokenType.BLOCK_DEF
                elif value.startswith("Block_"):
                    token_type = TokenType.BLOCK
                elif value.startswith("OPAQUE_"):
                    token_type = TokenType.OPAQUE_CONST
                elif value.startswith("MEM_"):
                    token_type = TokenType.MEMORY_OPERAND
                elif value.startswith("REG_LIST_"):
                    token_type = TokenType.REGISTER_LIST

                token_types.append(token_type)

                # Track Lit_Start and Lit_End tokens
                if "_LIT_START" in value.upper():
                    lit_start_tokens.append(index)

                if "_LIT_END" in value.upper():
                    lit_end_tokens.append(index)

            # Convert to numpy arrays
            v_man._id_to_token_type = np.array(token_types, dtype=np.int_)
            v_man._lit_start_cache = np.array(lit_start_tokens, dtype=np.int_)
            v_man._lit_start_count = len(lit_start_tokens)
            v_man._lit_end_cache = np.array(lit_end_tokens, dtype=np.int_)
            v_man._lit_end_count = len(lit_end_tokens)

        return v_man

    def _private_add_token(
        self,
        token: str,
        token_cls: type[Tokens],
        lit_type: LitTokenType = LitTokenType.REGULAR,
        insn_type: PlatformInstructionTypes = PlatformInstructionTypes.AGNOSTIC,
        platform: str = None,
    ) -> int:
        """Add a token to the vocabulary and return its ID, optionally setting platform instruction type."""
        if token in self.token_to_id:
            return self.token_to_id[token]

        assert (not (token.startswith("Block") or token.startswith("OPAQUE_CONST"))) or (
            token[-2] == "_" or "Lit" in token or token == "Block_Def"
        ), f"Warning: two digit token thats shouldnt: {token}"

        # Under v2 (per-binary CSV) and v1 (unified vocab) the
        # constructor pre-populates `id_to_token[0..255]` with
        # `digit_<HH>` placeholders so `self.size` is already >= 256
        # here — the ID-skip is automatic, no manual bump needed. Guard
        # against callers accidentally registering a token literally named
        # `digit_XX` which would shadow a digit slot. v1 unified carries
        # the same reserved-digit layout as v2 (see plan
        # memoized-booping-wren.md), so the collision check applies
        # unchanged.
        assert not (
            self.format_version in (1, 2)
            and len(token) == 8
            and token.startswith("digit_")
            and all(c in "0123456789ABCDEFabcdef" for c in token[6:])
        ), f"Cannot register token with reserved digit-slot name: {token}"

        # Add new token
        token_id = self.size
        self.token_to_id[token] = token_id
        self.id_to_token.append(token)

        # Get token type directly from the token class
        token_type = token_cls.token_type

        # Check if we need to expand token type capacity
        if token_id >= len(self._id_to_token_type):
            # Double the capacity
            old_capacity = len(self._id_to_token_type)
            new_capacity = old_capacity * 2

            # Resize id_to_token_type array
            new_token_type_array = np.empty(new_capacity, dtype=np.int8)
            new_platform_instruction_type_cache = np.full(
                new_capacity, PlatformInstructionTypes.AGNOSTIC, dtype=np.int8
            )
            new_token_type_array[:old_capacity] = self._id_to_token_type[:old_capacity]
            new_platform_instruction_type_cache[:old_capacity] = self._platform_instruction_type_cache[:old_capacity]
            self._id_to_token_type = new_token_type_array
            self._platform_instruction_type_cache = new_platform_instruction_type_cache
            if self.platform is None:
                new_platform_array = np.full(new_capacity, -1, dtype=np.int8)
                new_platform_array[:old_capacity] = self.token_to_platform[:old_capacity]
                self.token_to_platform = new_platform_array

        # Set token type
        self._id_to_token_type[token_id] = token_type
        self._platform_instruction_type_cache[token_id] = insn_type

        # handle platform
        if platform is not None:  # some token are platform-agnostic like Block_Def
            if self.platform is None:
                platform_id = self.platform_reverse.get(platform, -1)
                if platform_id == -1:
                    # Add new platform if it doesn't exist
                    platform_id = len(self.platform_list)
                    self.platform_list.append(platform)
                    self.platform_reverse[platform] = platform_id

                self.token_to_platform[token_id] = platform_id
            else:
                assert self.platform == platform, f"Platform mismatch: {self.platform} != {platform}"

        # Handle lit cache entries - only add if it's a lit token
        if lit_type == LitTokenType.LIT_START:
            # Expand lit_start_cache if needed
            if self._lit_start_count >= len(self._lit_start_cache):
                old_capacity = len(self._lit_start_cache)
                new_capacity = old_capacity * 2
                new_cache = np.empty(new_capacity, dtype=np.int_)
                new_cache[:old_capacity] = self._lit_start_cache[:old_capacity]
                self._lit_start_cache = new_cache

            self._lit_start_cache[self._lit_start_count] = token_id
            self._lit_start_count += 1

        elif lit_type == LitTokenType.LIT_END:
            # Expand lit_end_cache if needed
            if self._lit_end_count >= len(self._lit_end_cache):
                old_capacity = len(self._lit_end_cache)
                new_capacity = old_capacity * 2
                new_cache = np.empty(new_capacity, dtype=np.int_)
                new_cache[:old_capacity] = self._lit_end_cache[:old_capacity]
                self._lit_end_cache = new_cache

            self._lit_end_cache[self._lit_end_count] = token_id
            self._lit_end_count += 1

        # Regular tokens don't get added to lit caches at all
        return token_id

    @property
    def id_to_token_type(self) -> npt.NDArray[np.int8]:
        """Get readonly view of id_to_token_type array"""
        result = self._id_to_token_type[: self.size].view()
        result.flags.writeable = False
        return result

    @property
    def lit_starts(self) -> npt.NDArray[np.int_]:
        """Get readonly view of lit_start_cache array"""
        result = self._lit_start_cache[: self._lit_start_count].view()
        result.flags.writeable = False
        return result

    @property
    def lit_ends(self) -> npt.NDArray[np.int_]:
        """Get readonly view of lit_end_cache array"""
        result = self._lit_end_cache[: self._lit_end_count].view()
        result.flags.writeable = False
        return result

    def get_registry_token(self, reg_name: str, reg_id: int) -> Tokens:
        if len(self.registry_token_cache) <= reg_id:
            # Ensure the list is large enough
            self.registry_token_cache.extend([None] * (reg_id - len(self.registry_token_cache) + 1))

        if self.registry_token_cache[reg_id] is None:
            token = self.PlatformToken(reg_name, insn_type=PlatformInstructionTypes.REGISTRY)
            self.registry_token_cache[reg_id] = token
        else:
            token = self.registry_token_cache[reg_id]
            assert str(token) == f"{self.platform}_{reg_name}", "Token mismatch for register ID"

        return token

    def get_token_id(self, token: str) -> int:
        """Get the ID for a token, or -1 if not found"""
        return self.token_to_id.get(token, -1)

    def get_token_str(self, token_id: int) -> str:
        """Get the token string for an ID, or empty string if not found"""
        if 0 <= token_id < len(self.id_to_token):
            return self.id_to_token[token_id]
        return ""

    @property
    def size(self) -> int:
        """Return the number of tokens in the vocabulary"""
        return len(self.id_to_token)

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary format for backward compatibility"""
        return self.token_to_id.copy()

    def create_token_from_insn_list(self, insn_token_list: "InsnTokenList", index: int) -> "Tokens":
        """Create a single token from an InsnTokenList at the specified index"""
        if insn_token_list.last_index == 0:
            raise IndexError("Cannot get token from empty instruction token list")

        if index < 0 or index >= insn_token_list.last_index:
            raise IndexError(f"Token index {index} out of bounds (0 to {insn_token_list.last_index - 1})")
        if insn_token_list.metatoken_start_lookup is None:
            raise ValueError("Cannot get token from invalidated view")

        token_type = TokenType(insn_token_list.metatoken_type_ids[index])

        # Get token IDs for this specific token
        start_pos = insn_token_list.metatoken_start_lookup[index - 1] if index > 0 else 0
        if index == insn_token_list.last_index - 1:
            end_pos = len(insn_token_list.get_used_token_ids())
        else:
            end_pos = insn_token_list.metatoken_start_lookup[index]

        token_ids = insn_token_list.token_ids[start_pos:end_pos].tolist()
        return self._reconstruct_token_from_ids(token_type, token_ids)

    def get_token_class_for_type(self, token_type: TokenType) -> type[Tokens]:
        """Get the token class for a given token type"""
        if token_type == TokenType.PLATFORM:
            return self.PlatformToken
        elif token_type == TokenType.VALUED_CONST:
            return self.Valued_Const
        elif token_type == TokenType.BLOCK_DEF:
            return self.Block_Def
        elif token_type == TokenType.BLOCK:
            return self.Block
        elif token_type == TokenType.OPAQUE_CONST:
            return self.Opaque_Const
        elif token_type == TokenType.MEMORY_OPERAND:
            return self.MemoryOperand
        elif token_type == TokenType.REGISTER_LIST:
            return self.RegisterList
        elif token_type == TokenType.TOKEN_SET:
            return self.TokenSet
        # v2 category tokens (plan vivid-tinkering-wilkes.md). Dispatch
        # table parallels the registration table at the bottom of
        # `_create_inner_classes`.
        elif token_type == TokenType.LOCAL_FUNC:
            return self.Local_Func
        elif token_type == TokenType.PLT_FUNC:
            return self.Plt_Func
        elif token_type == TokenType.EXT_FUNC:
            return self.Ext_Func
        elif token_type == TokenType.RO_DATA_PTR:
            return self.Ro_Data_Ptr
        elif token_type == TokenType.RW_DATA_PTR:
            return self.Rw_Data_Ptr
        elif token_type == TokenType.STRING_PTR:
            return self.String_Ptr
        elif token_type == TokenType.JUMP_TABLE:
            return self.Jump_Table
        elif token_type == TokenType.BLOCK_V2:
            return self.Block_V2
        elif token_type == TokenType.VALUED_CONST_V2:
            return self.Valued_Const_V2
        elif token_type == TokenType.FLOAT16:
            return self.Float16
        elif token_type == TokenType.BFLOAT16:
            return self.BFloat16
        elif token_type == TokenType.FLOAT32:
            return self.Float32
        elif token_type == TokenType.FLOAT64:
            return self.Float64
        elif token_type == TokenType.FLOAT80:
            return self.Float80
        elif token_type == TokenType.FLOAT128:
            return self.Float128
        elif token_type == TokenType.THREAD_LOCAL:
            return self.Thread_Local
        elif token_type == TokenType.VTABLE:
            return self.Vtable
        elif token_type == TokenType.CODE_PTR_TABLE:
            return self.Code_Ptr_Table
        elif token_type == TokenType.VARIANT_AXIS:
            return self.Variant_Axis
        else:
            raise ValueError(f"Unknown token type: {token_type}")

    def _reconstruct_token_from_ids(self, token_type: TokenType, token_ids: List[int]) -> "Tokens":
        """Reconstruct a token from its type and token IDs"""
        token_class = self.get_token_class_for_type(token_type)
        return token_class._from_token_ids(token_ids)

    # Format-aware factory dispatchers. Consumer code that needs to emit a
    # valued constant or block identifier in the active wire format should
    # call these instead of branching on `format_version` at the call site
    # (single-source-of-truth for v1-unified / v2-per-binary-CSV dispatch;
    # the inline-digit encoding is shared between the two). Variant axes
    # are additive in v1, registered separately via the Variant_Axis Inner
    # class.
    def ValuedConst(self, value):
        return self.Valued_Const_V2(value) if self.format_version in (1, 2) else self.Valued_Const(value)

    def BlockId(self, block_id):
        return self.Block_V2(block_id) if self.format_version in (1, 2) else self.Block(block_id)

    def iter_representative_tokens(self):
        identifier_token_ids = []
        valued_const_ids = []
        lit_starts = set(self.lit_starts.tolist())
        lit_ends = set(self.lit_ends.tolist())
        lits = lit_starts.union(lit_ends)
        self.Valued_Const._token_ids = np.array([], dtype=np.int_)
        # Under format_version=1 (unified vocab) and =2 (per-binary CSV)
        # the first `_V2_RESERVED_DIGIT_COUNT` IDs are protocol-reserved
        # digit slots (carrying TokenType.UNRESOLVED). They are not real
        # vocab entries — no registration, no string, no representative
        # token. Skip them so the dispatch table never sees the
        # placeholder UNRESOLVED type. v1 unified reuses the v2
        # reserved-digit layout (variant tokens are additive at IDs
        # 256+); the inline-digit representative path below also applies
        # unchanged.
        v2 = self.format_version in (1, 2)
        start_id = self._V2_RESERVED_DIGIT_COUNT if v2 else 0
        for i in range(start_id, self.size):
            if i in lits:
                continue

            token_type = self._id_to_token_type[i]
            if token_type == TokenType.IDENTIFIER_LITERAL:
                identifier_token_ids.append(i)
                continue
            elif token_type == TokenType.VALUED_CONST:
                valued_const_ids.append(i)

            if v2:
                v2_token = self._make_v2_representative(token_type)
                if v2_token is not None:
                    yield v2_token
                    continue

            yield self._reconstruct_token_from_ids(token_type, [i])

        lit_starts = {self._id_to_token_type[id]: id for id in lit_starts}
        lits = {self._id_to_token_type[end]: (lit_starts[self._id_to_token_type[end]], end) for end in lit_ends}

        if TokenType.VALUED_CONST in lits:
            vs, ve = lits[TokenType.VALUED_CONST]
            # valued_const_ids are already yielded as singletons, so we can just use the first three (not two as the first could be 00)
            yield self._reconstruct_token_from_ids(TokenType.VALUED_CONST, [vs] + valued_const_ids[:3] + [ve])

        # we must include all identifier tokens even if we do not have opaque or block tokens
        has_opaque = TokenType.OPAQUE_CONST in lits
        has_blocks = TokenType.BLOCK in lits
        identifier_token_opaque = identifier_token_ids[:3] + (
            identifier_token_ids[3 : len(identifier_token_ids) // 2] if has_blocks else identifier_token_ids[3:]
        )
        identifier_token_blocks = identifier_token_ids[:3] + (
            identifier_token_ids[len(identifier_token_ids) // 2 :] if has_opaque else identifier_token_ids[3:]
        )

        if has_opaque:
            os, oe = lits[TokenType.OPAQUE_CONST]
            yield self._reconstruct_token_from_ids(TokenType.OPAQUE_CONST, [os] + identifier_token_opaque + [oe])

        if has_blocks:
            bs, be = lits[TokenType.BLOCK]
            yield self._reconstruct_token_from_ids(TokenType.BLOCK, [bs] + identifier_token_blocks + [be])

    # v2 category tokens cannot be reconstructed from a single type-id via
    # `_reconstruct_token_from_ids` because their `_from_token_ids` asserts
    # the full wire shape (`>= 2` ids for identity/valued_const_v2, exact
    # `1` or `1 + width_bytes` for floats). For the unifier's representative
    # iteration we only need ONE instance per registered type so the
    # remap-table builder can resolve `mappings[type_id] = unified_type_id`;
    # the identity payload is irrelevant to that mapping.
    #
    # Construct a minimal-payload representative per category instead:
    #   * identity tokens and valued_const_v2 → `cls(0)` (1 digit byte for 0)
    #   * float tokens → `cls(None)` (postfix-annotation form: type id only)
    #   * modifier tokens → `cls()` (no payload)
    #
    # Returns `None` for non-v2 token types so the caller falls back to the
    # legacy `_reconstruct_token_from_ids` path (PLATFORM, BLOCK_DEF,
    # MEMORY_OPERAND, etc. all still work as singletons under v2 too).
    # VARIANT_AXIS (v1-unified-vocab-only opaque-string family) also takes the singleton
    # fallback: each vocab id is a real registered string and `Variant_Axis
    # ._from_token_ids([id])` simply looks up the string from the vocab.
    # No representative-shape collapse is meaningful because each id IS its
    # own distinct token (string is the data, not a digit payload).
    _V2_IDENTITY_TOKEN_TYPES = frozenset({
        TokenType.LOCAL_FUNC,
        TokenType.PLT_FUNC,
        TokenType.EXT_FUNC,
        TokenType.RO_DATA_PTR,
        TokenType.RW_DATA_PTR,
        TokenType.STRING_PTR,
        TokenType.JUMP_TABLE,
        TokenType.BLOCK_V2,
    })
    _V2_FLOAT_TOKEN_TYPES = frozenset({
        TokenType.FLOAT16,
        TokenType.BFLOAT16,
        TokenType.FLOAT32,
        TokenType.FLOAT64,
        TokenType.FLOAT80,
        TokenType.FLOAT128,
    })
    _V2_MODIFIER_TOKEN_TYPES = frozenset({
        TokenType.THREAD_LOCAL,
        TokenType.VTABLE,
        TokenType.CODE_PTR_TABLE,
    })

    def _make_v2_representative(self, token_type: TokenType):
        """Return a single representative Tokens instance for a v2 category,
        or None if `token_type` is not a v2 category (caller falls back to
        the legacy singleton dispatch)."""
        token_cls = self.get_token_class_for_type(token_type)
        if token_type in self._V2_IDENTITY_TOKEN_TYPES:
            return token_cls(0)
        if token_type == TokenType.VALUED_CONST_V2:
            return token_cls(0)
        if token_type in self._V2_FLOAT_TOKEN_TYPES:
            # Postfix-annotation form: bits=None ⇒ wire = `[type_id]`.
            return token_cls(None)
        if token_type in self._V2_MODIFIER_TOKEN_TYPES:
            return token_cls()
        return None

    def _create_inner_classes(self):
        """Create inner classes that have access to this VocabularyManager instance"""
        vocab_manager = self  # Capture the instance

        class TokensInner(Tokens, ABC):
            """Abstract base class for all token representations"""

            @abstractmethod
            def get_token_ids(self) -> npt.NDArray[np.int_]:
                """Get the list of token IDs for this token representation (order matters)"""
                pass

            @abstractmethod
            def to_string(self) -> str:
                """Convert token to its string representation (for debugging only)"""
                pass

        # Ensure TokensInner conforms to Tokens protocol
        assert issubclass(TokensInner, Tokens)

        class PlatformTokenInner(TokensInner, PlatformToken):
            """Represents platform-specific tokens like x86 instructions, registers, etc.

            In a per-binary VocabularyManager (`vocab_manager.platform`
            is set to a single ISA), tokens are stored under
            `<platform>_<token>` — the per-ISA namespace. In a unified
            VocabularyManager (`vocab_manager.platform is None`), tokens
            are stored under `<family>_<token>` so cross-bitness siblings
            (e.g. mips32 + mips64 both producing `addu`) collapse to a
            single ID. The original ISA each occurrence came from is
            still tracked via `token_to_platform[token_id]`, so
            consumers that need per-bitness provenance haven't lost
            information.
            """

            __slots__ = ("token", "_token_id")

            def __init__(self, token: str, insn_type: PlatformInstructionTypes, platform: str = None):
                if " " in token:
                    raise ValueError(f"Token cannot contain spaces: '{token}'")
                if platform is None:
                    platform = vocab_manager.platform

                # Family-merge prefix when registering on a unified VM;
                # per-ISA prefix when on a per-binary VM. The lookup
                # prefers PLATFORM_FAMILY but falls back to the
                # platform name itself if the platform isn't in the
                # family map (test fixtures, custom ISAs).
                from tokenizer.arch import PLATFORM_FAMILY
                if vocab_manager.platform is None:
                    name_prefix = PLATFORM_FAMILY.get(platform, platform)
                else:
                    name_prefix = platform

                self.token = token
                # `platform=` retains the original per-ISA name so
                # `_private_add_token`'s platform_list / token_to_platform
                # tracking records which specific ISA contributed this
                # token. The token NAME uses the family prefix when
                # registering on a unified VM.
                self._token_id = vocab_manager._private_add_token(
                    f"{name_prefix}_{token}", self.__class__,
                    insn_type=insn_type, platform=platform,
                )

            @classmethod
            def _from_token_ids(cls, token_ids: List[int]) -> "PlatformTokenInner":
                """Reconstruct a PlatformToken from token IDs"""
                if len(token_ids) != 1:
                    raise ValueError(f"Platform token must have exactly one ID, got {len(token_ids)}")

                token_str = vocab_manager.get_token_str(token_ids[0])
                platform = (
                    vocab_manager.platform_list[vocab_manager.token_to_platform[token_ids[0]]]
                    if vocab_manager.platform is None
                    else vocab_manager.platform
                )
                # On a unified VM the token's stored prefix is the
                # FAMILY (e.g. `mips`), not the per-ISA platform
                # (`mips32` / `mips64`). Resolve the expected prefix
                # accordingly.
                from tokenizer.arch import PLATFORM_FAMILY
                if vocab_manager.platform is None:
                    expected_prefix = PLATFORM_FAMILY.get(platform, platform)
                else:
                    expected_prefix = platform
                if not token_str.startswith(f"{expected_prefix}_"):
                    raise ValueError(f"Invalid platform token string: {token_str}")

                platform_token = token_str[len(expected_prefix) + 1 :]
                return cls(
                    platform_token, vocab_manager._platform_instruction_type_cache[token_ids[0]], platform=platform
                )

            def get_token_ids(self) -> npt.NDArray[np.int_]:
                return np.array([self._token_id], dtype=np.int_)

            def to_string(self) -> str:
                return f"{vocab_manager.platform}_{self.token}"

            def to_asm_like(self) -> str:
                return self.token

            @property
            def platform_instruction_type(self) -> PlatformInstructionTypes:
                """Get the platform instruction type for this token"""
                return PlatformInstructionTypes(vocab_manager._platform_instruction_type_cache[self._token_id])

            @property
            def platform(self) -> str:
                """Get the platform for this token"""
                return vocab_manager.platform

        # Ensure PlatformTokenInner conforms to both protocols
        assert issubclass(PlatformTokenInner, Tokens)
        assert issubclass(PlatformTokenInner, PlatformToken)

        class ValuedConstTokenInner(TokensInner, ValuedConstToken):
            """Represents a constant with a specific numeric value"""

            __slots__ = ("value", "_token_ids")

            def __init__(self, value: int):
                self.value = value

                # Handle negative values
                is_negative = value < 0
                abs_value = abs(value)

                # Generate hex string with proper padding
                hex_str = f"{abs_value:02X}"  # Always at least 2 digits, uppercase
                if len(hex_str) % 2 == 1:
                    hex_str = "0" + hex_str  # Pad to even length

                # Convert hex string chunks to integer values
                hex_values = [int(hex_str[i : i + 2], 16) for i in range(0, len(hex_str), 2)]
                hex_values_array = np.array(hex_values, dtype=np.int_)
                self._token_ids = TokenUtils.encode_tokens(
                    "VALUED_CONST",
                    "VALUED_CONST",
                    hex_values_array,
                    vocab_manager,
                    token_class=self.__class__,
                    inner_token_class=self.__class__,
                    max_key=256,
                    include_minus=is_negative,
                )

            @classmethod
            def _from_token_ids(cls, token_ids: List[int]) -> "ValuedConstTokenInner":
                """Reconstruct a ValuedConstToken from token IDs using utility method"""
                value = TokenUtils.decode_tokens_to_value(
                    token_ids,
                    "VALUED_CONST",
                    "VALUED_CONST",
                    vocab_manager,
                    max_key=256,
                    support_negative=True,
                    token_class=cls,
                    inner_token_class=cls,
                )
                return cls(value)

            def get_token_ids(self) -> npt.NDArray[np.int_]:
                return np.array(self._token_ids, dtype=np.int_)

            def to_string(self) -> str:
                """Generate string representation for debugging"""
                is_negative = self.value < 0
                abs_value = abs(self.value)
                hex_str = f"{abs_value:02X}"  # Always at least 2 digits, uppercase

                if abs_value <= 0xFF and not is_negative:
                    return f"VALUED_CONST_{hex_str}"
                else:
                    # Multi-token representation or negative value
                    if len(hex_str) % 2 == 1:
                        hex_str = "0" + hex_str

                    chunks = [hex_str[i : i + 2] for i in range(0, len(hex_str), 2)]

                    name = "VALUED_CONST_Lit_Start"
                    if is_negative:
                        name += " MEM_MINUS"
                    for chunk in chunks:
                        name += f" VALUED_CONST_{chunk}"
                    name += " VALUED_CONST_Lit_End"
                    return name

            def to_asm_like(self) -> str:
                return f"v:{self.value:x}"

        # Ensure ValuedConstTokenInner conforms to both protocols
        assert issubclass(ValuedConstTokenInner, Tokens)
        assert issubclass(ValuedConstTokenInner, ValuedConstToken)

        class IdentifierInner(TokensInner, IdentifierToken, ABC):
            """Abstract base class for identifiers with IDs"""

            __slots__ = ("id", "_token_ids")

            def __init__(self, identifier_id: int):
                IdentifierToken.__init__(self, identifier_id)
                TokensInner.__init__(self)
                self.id = identifier_id

                basename = self._get_basename()

                hex_str = f"{self.id:X}"
                # Convert hex string characters to integer values
                hex_values = [int(c, 16) for c in hex_str]
                hex_values_array = np.array(hex_values, dtype=np.int_)
                self._token_ids = TokenUtils.encode_tokens(
                    basename,
                    "Identifier_Lit",
                    hex_values_array,
                    vocab_manager,
                    token_class=self.__class__,
                    inner_token_class=IdentifierInner,
                    max_key=16,
                )

            @classmethod
            def singleton_token_index(cls, id: int) -> typing.Optional[int]:
                """Get the index of a singleton token with a single hex digit"""
                if 0 <= id < 16:
                    result = TokenUtils.cache_numeric_token(cls, cls._get_basename(), id, lambda: -1, max_key=16)
                    if result >= 0:
                        return result

                return None

            @classmethod
            def value_by_singleton_token_index(cls, index: int) -> typing.Optional[int]:
                result = TokenUtils.cache_numeric_reverse(cls, index, cls._get_basename(), vocab_manager)
                if result >= 0:
                    return result
                return None

            @classmethod
            def _from_token_ids(cls, token_ids: List[int]) -> "IdentifierInner":
                """Reconstruct an IdentifierToken from token IDs using utility method"""
                identifier_id = TokenUtils.decode_tokens_to_value(
                    token_ids,
                    cls._get_basename(),
                    "Identifier_Lit",
                    vocab_manager,
                    max_key=16,
                    support_negative=False,
                    token_class=cls,
                    inner_token_class=IdentifierInner,
                )

                return cls(identifier_id)

            def get_token_ids(self) -> npt.NDArray[np.int_]:
                return np.array(self._token_ids, dtype=np.int_)

            def to_string(self) -> str:
                """Generate string representation for debugging (recreates register_name_range output)"""
                basename = self._get_basename()
                if self.id < 16:
                    hex_str = f"{self.id:X}"
                    return f"{basename}_{hex_str}"
                else:
                    id_str = f"{self.id:X}"
                    name = f"{basename}_Lit_Start"
                    for hex_digit in id_str:
                        name += f" Identifier_Lit_{hex_digit}"
                    name += f" {basename}_Lit_End"
                    return name

        # Ensure IdentifierInner conforms to both protocols
        assert issubclass(IdentifierInner, Tokens)
        assert issubclass(IdentifierInner, IdentifierToken)

        class BlockDefInner(TokensInner, BlockDefToken):
            """Represents block definition tokens (Block_Def)"""

            __slots__ = ("_token_id",)

            def __init__(self):
                # Register the token and cache its ID
                self._token_id = vocab_manager._private_add_token("Block_Def", self.__class__)

            @classmethod
            def _from_token_ids(cls, token_ids: List[int]) -> "BlockDefInner":
                """Reconstruct a BlockDefToken from token IDs"""
                if len(token_ids) != 1:
                    raise ValueError(f"Block def token must have exactly one ID, got {len(token_ids)}")

                token_str = vocab_manager.get_token_str(token_ids[0])
                if token_str != "Block_Def":
                    raise ValueError(f"Invalid block def token string: {token_str}")

                return cls()

            def get_token_ids(self) -> npt.NDArray[np.int_]:
                return np.array([self._token_id], dtype=np.int_)

            def to_string(self) -> str:
                return "Block_Def"

            def to_asm_like(self) -> str:
                return "_def"

        # Ensure BlockDefInner conforms to both protocols
        assert issubclass(BlockDefInner, Tokens)
        assert issubclass(BlockDefInner, BlockDefToken)

        class BlockInner(IdentifierInner, BlockToken):
            """Represents block identifiers"""

            __slots__ = ()

            def __init__(self, block_id: int):
                super().__init__(block_id)

            @classmethod
            def _get_basename(cls) -> str:
                return "Block"

            def to_asm_like(self) -> str:
                return f"block:{self.id}"

        # Ensure BlockInner conforms to both protocols
        assert issubclass(BlockInner, IdentifierToken)
        assert issubclass(BlockInner, BlockToken)

        class OpaqueConstInner(IdentifierInner, OpaqueConstToken):
            """Represents opaque constant identifiers"""

            __slots__ = ()

            def __init__(self, opaque_id: int):
                super().__init__(opaque_id)

            @classmethod
            def _get_basename(cls) -> str:
                return "OPAQUE_CONST"

            def to_asm_like(self) -> str:
                return f"opaque:{self.id}"

        # Ensure OpaqueConstInner conforms to both protocols
        assert issubclass(OpaqueConstInner, IdentifierToken)
        assert issubclass(OpaqueConstInner, OpaqueConstToken)

        class LocalFunctionInner(IdentifierInner, LocalFunctionToken):
            """Represents local function identifiers - used by inlining matcher, not by tokenizer"""

            __slots__ = ()

            def __init__(self, local_function_id: int):
                super().__init__(local_function_id)

            @classmethod
            def _get_basename(cls) -> str:
                return "LOCAL_FUNCTION"

            def to_asm_like(self) -> str:
                return f"local_fn:{self.id}"

        # Ensure LocalFunctionInner conforms to both protocols
        assert issubclass(LocalFunctionInner, IdentifierToken)
        assert issubclass(LocalFunctionInner, LocalFunctionToken)

        # ------------------------------------------------------------------
        # v2 inline-digit Inner classes (plan vivid-tinkering-wilkes.md).
        #
        # Design boundary: every v2 Inner class lives in this nest, owns
        # exactly one vocab entry (registered on first instance), and emits
        # `[type_token_id, *digit_bytes]` where digit bytes are raw uint8
        # values that double as vocab ids 0..255 (reserved at construction
        # in `__init__`). The encoding shape is identical across categories
        # — the only per-category data is the basename, so the encoding
        # logic lives ONCE on the abstract `_V2IdentityInner` /
        # `_V2FloatInner` / `_V2ModifierInner` mixins below; concrete
        # subclasses are 3-line glue (basename + ABC binding + `to_asm_like`).
        #
        # Decoding mirrors the same one-place rule: each mixin's
        # `_from_token_ids` strips the type-id and reassembles the payload.
        #
        # All v2 Inner classes assert `vocab_manager.format_version in
        # (1, 2)` on instantiation to enforce the reserved-id-protocol
        # invariant — a VM without reserved digit slots would silently
        # collide with low-id legacy entries. v1 (the unified vocab; see
        # plan memoized-booping-wren.md) reuses the v2 reserved-digit
        # layout, so the same Inner classes are valid on a v1 VM and
        # produce identical wire streams. Variant tokens are additive at
        # IDs 256+ on v1.
        # ------------------------------------------------------------------

        class _V2IdentityInner(TokensInner, IdentifierToken, ABC):
            """Mixin for v2 identity-carrying category tokens.

            Concrete subclasses bind a v2 ABC (`LocalFuncToken`, `PltFuncToken`,
            ...) and supply `_get_basename`. The vocab basename strings are
            the literal lowercase names from the plan (e.g., `local_func`,
            `block_v2`) — distinct from v1 names so v1 and v2 entries can
            coexist in a unified vocab when the unifier merges per-binary
            outputs in a mixed-version corpus.
            """

            __slots__ = ("id", "_token_ids", "_type_token_id")

            def __init__(self, identifier_id: int):
                # IdentifierToken's __init__ stores `id`; replicate that
                # contract without delegating to legacy `IdentifierInner`'s
                # encode_tokens path.
                assert vocab_manager.format_version in (1, 2), (
                    "v2 Inner classes require format_version=1 (unified) or =2 (per-binary CSV) VocabularyManager; "
                    f"got format_version={vocab_manager.format_version}"
                )
                assert identifier_id >= 0, f"v2 identity must be non-negative, got {identifier_id}"
                self.id = identifier_id

                basename = self._get_basename()
                self._type_token_id = vocab_manager._private_add_token(basename, self.__class__)
                payload = _v2_int_to_minimum_bytes(identifier_id)
                self._token_ids = [self._type_token_id, *payload]

            @classmethod
            def _from_token_ids(cls, token_ids: List[int]) -> "_V2IdentityInner":
                """Reconstruct a v2 identity token from its `[type_id, *digits]` slice."""
                if len(token_ids) < 2:
                    raise ValueError(
                        f"v2 identity token must have >=2 ids (type + >=1 digit), got {token_ids}"
                    )
                identifier_id = _v2_bytes_to_int(token_ids[1:])
                return cls(identifier_id)

            def get_token_ids(self) -> npt.NDArray[np.int_]:
                return np.array(self._token_ids, dtype=np.int_)

            def to_string(self) -> str:
                # Debug string mirrors the wire format: basename followed
                # by hex digits of each payload byte.
                basename = self._get_basename()
                digits = " ".join(f"digit_{b:02X}" for b in self._token_ids[1:])
                return f"{basename} {digits}" if digits else basename

        # Concrete v2 identity Inner classes — one per category. Each is a
        # thin glue that picks a basename + binds its v2 ABC + renders
        # `to_asm_like`. The encoding/decoding logic lives in
        # `_V2IdentityInner` (single-concern rule).

        class LocalFuncInner(_V2IdentityInner, LocalFuncToken):
            __slots__ = ()

            def __init__(self, local_func_id: int):
                super().__init__(local_func_id)

            @classmethod
            def _get_basename(cls) -> str:
                return "local_func"

            def to_asm_like(self) -> str:
                return f"local_func:{self.id}"

        assert issubclass(LocalFuncInner, IdentifierToken)
        assert issubclass(LocalFuncInner, LocalFuncToken)

        class PltFuncInner(_V2IdentityInner, PltFuncToken):
            __slots__ = ()

            def __init__(self, plt_func_id: int):
                super().__init__(plt_func_id)

            @classmethod
            def _get_basename(cls) -> str:
                return "plt_func"

            def to_asm_like(self) -> str:
                return f"plt_func:{self.id}"

        assert issubclass(PltFuncInner, IdentifierToken)
        assert issubclass(PltFuncInner, PltFuncToken)

        class ExtFuncInner(_V2IdentityInner, ExtFuncToken):
            __slots__ = ()

            def __init__(self, ext_func_id: int):
                super().__init__(ext_func_id)

            @classmethod
            def _get_basename(cls) -> str:
                return "ext_func"

            def to_asm_like(self) -> str:
                return f"ext_func:{self.id}"

        assert issubclass(ExtFuncInner, IdentifierToken)
        assert issubclass(ExtFuncInner, ExtFuncToken)

        class RoDataPtrInner(_V2IdentityInner, RoDataPtrToken):
            __slots__ = ()

            def __init__(self, ro_data_ptr_id: int):
                super().__init__(ro_data_ptr_id)

            @classmethod
            def _get_basename(cls) -> str:
                return "ro_data_ptr"

            def to_asm_like(self) -> str:
                return f"ro_data_ptr:{self.id}"

        assert issubclass(RoDataPtrInner, IdentifierToken)
        assert issubclass(RoDataPtrInner, RoDataPtrToken)

        class RwDataPtrInner(_V2IdentityInner, RwDataPtrToken):
            __slots__ = ()

            def __init__(self, rw_data_ptr_id: int):
                super().__init__(rw_data_ptr_id)

            @classmethod
            def _get_basename(cls) -> str:
                return "rw_data_ptr"

            def to_asm_like(self) -> str:
                return f"rw_data_ptr:{self.id}"

        assert issubclass(RwDataPtrInner, IdentifierToken)
        assert issubclass(RwDataPtrInner, RwDataPtrToken)

        class StringPtrInner(_V2IdentityInner, StringPtrToken):
            __slots__ = ()

            def __init__(self, string_ptr_id: int):
                super().__init__(string_ptr_id)

            @classmethod
            def _get_basename(cls) -> str:
                return "string_ptr"

            def to_asm_like(self) -> str:
                return f"string_ptr:{self.id}"

        assert issubclass(StringPtrInner, IdentifierToken)
        assert issubclass(StringPtrInner, StringPtrToken)

        class JumpTableInner(_V2IdentityInner, JumpTableToken):
            __slots__ = ()

            def __init__(self, jump_table_id: int):
                super().__init__(jump_table_id)

            @classmethod
            def _get_basename(cls) -> str:
                return "jump_table"

            def to_asm_like(self) -> str:
                return f"jump_table:{self.id}"

        assert issubclass(JumpTableInner, IdentifierToken)
        assert issubclass(JumpTableInner, JumpTableToken)

        class BlockV2Inner(_V2IdentityInner, BlockTokenV2):
            __slots__ = ()

            def __init__(self, block_id: int):
                super().__init__(block_id)

            @classmethod
            def _get_basename(cls) -> str:
                return "block_v2"

            def to_asm_like(self) -> str:
                return f"block_v2:{self.id}"

        assert issubclass(BlockV2Inner, IdentifierToken)
        assert issubclass(BlockV2Inner, BlockTokenV2)

        # ValuedConstV2 — same wire shape as identity tokens (minimum-width
        # big-endian payload), but the ABC exposes `value` instead of `id`,
        # so it can't share `_V2IdentityInner`. Implementation parallels it
        # to avoid duplicating the encoding rule: payload bytes come from
        # `_v2_int_to_minimum_bytes` and decoding from `_v2_bytes_to_int`.

        class ValuedConstV2Inner(TokensInner, ValuedConstTokenV2):
            """Represents a v2 valued-constant with variable-width inline payload."""

            __slots__ = ("value", "_token_ids", "_type_token_id")

            def __init__(self, value: int):
                assert vocab_manager.format_version in (1, 2), (
                    "v2 Inner classes require format_version=1 (unified) or =2 (per-binary CSV) VocabularyManager; "
                    f"got format_version={vocab_manager.format_version}"
                )
                # Plan reserves negative-value semantics; current impl
                # restricts to non-negative until the ConstantHandler
                # rewrite (Phase 1.C.1) settles on a two's-complement vs.
                # MEM_MINUS-prefix vs. sign-byte choice. Failing fast here
                # surfaces the gap rather than silently emitting a corrupt
                # stream.
                assert value >= 0, (
                    f"v2 valued_const negative-value encoding is not yet specified; got {value}"
                )
                self.value = value
                self._type_token_id = vocab_manager._private_add_token("valued_const_v2", self.__class__)
                payload = _v2_int_to_minimum_bytes(value)
                self._token_ids = [self._type_token_id, *payload]

            @classmethod
            def _from_token_ids(cls, token_ids: List[int]) -> "ValuedConstV2Inner":
                if len(token_ids) < 2:
                    raise ValueError(
                        f"v2 valued_const token must have >=2 ids (type + >=1 digit), got {token_ids}"
                    )
                value = _v2_bytes_to_int(token_ids[1:])
                return cls(value)

            def get_token_ids(self) -> npt.NDArray[np.int_]:
                return np.array(self._token_ids, dtype=np.int_)

            def to_string(self) -> str:
                digits = " ".join(f"digit_{b:02X}" for b in self._token_ids[1:])
                return f"valued_const_v2 {digits}" if digits else "valued_const_v2"

            def to_asm_like(self) -> str:
                return f"v2:{self.value:x}"

        assert issubclass(ValuedConstV2Inner, ValuedConstToken)
        assert issubclass(ValuedConstV2Inner, ValuedConstTokenV2)

        # Float Inner classes. Two-mode payload: `bits is None` emits only
        # the type id (postfix annotation form); `bits` set emits the type
        # id followed by exactly `width_bytes` big-endian digit bytes
        # (fixed width so the reader can consume the right count without
        # ambiguity). Width is per-subclass — taken from the ABC's
        # `width_bytes` classvar (already set on Float16Token/.../
        # Float128Token).

        class _V2FloatInner(TokensInner, FloatToken, ABC):
            """Mixin for v2 float-category tokens (fixed-width or postfix)."""

            __slots__ = ("bits", "_token_ids", "_type_token_id")

            def __init__(self, bits: Optional[int] = None):
                assert vocab_manager.format_version in (1, 2), (
                    "v2 Inner classes require format_version=1 (unified) or =2 (per-binary CSV) VocabularyManager; "
                    f"got format_version={vocab_manager.format_version}"
                )
                if bits is not None:
                    assert bits >= 0, f"float bits must be unsigned bit pattern, got {bits}"
                    max_bits = 1 << (self.width_bytes * 8)
                    assert bits < max_bits, (
                        f"float bits 0x{bits:x} does not fit in {self.width_bytes} bytes"
                    )
                self.bits = bits
                self._type_token_id = vocab_manager._private_add_token(self._get_basename(), self.__class__)
                if bits is None:
                    self._token_ids = [self._type_token_id]
                else:
                    payload = bits.to_bytes(self.width_bytes, "big")
                    self._token_ids = [self._type_token_id, *payload]

            @classmethod
            @abstractmethod
            def _get_basename(cls) -> str:
                ...

            @classmethod
            def _from_token_ids(cls, token_ids: List[int]) -> "_V2FloatInner":
                if len(token_ids) == 1:
                    return cls(None)
                if len(token_ids) - 1 != cls.width_bytes:
                    raise ValueError(
                        f"v2 {cls._get_basename()} expects 1 (postfix) or "
                        f"{1 + cls.width_bytes} (inline) ids, got {len(token_ids)}"
                    )
                bits = _v2_bytes_to_int(token_ids[1:])
                return cls(bits)

            def get_token_ids(self) -> npt.NDArray[np.int_]:
                return np.array(self._token_ids, dtype=np.int_)

            def to_string(self) -> str:
                basename = self._get_basename()
                if self.bits is None:
                    return basename
                digits = " ".join(f"digit_{b:02X}" for b in self._token_ids[1:])
                return f"{basename} {digits}"

            def to_asm_like(self) -> str:
                basename = self._get_basename()
                if self.bits is None:
                    return basename  # postfix annotation; value lives elsewhere
                return f"{basename}:{self.bits:0{self.width_bytes * 2}x}"

        class Float16Inner(_V2FloatInner, Float16Token):
            __slots__ = ()

            def __init__(self, bits: Optional[int] = None):
                super().__init__(bits)

            @classmethod
            def _get_basename(cls) -> str:
                return "float16"

        assert issubclass(Float16Inner, FloatToken)
        assert issubclass(Float16Inner, Float16Token)
        assert Float16Inner.width_bytes == 2

        class BFloat16Inner(_V2FloatInner, BFloat16Token):
            __slots__ = ()

            def __init__(self, bits: Optional[int] = None):
                super().__init__(bits)

            @classmethod
            def _get_basename(cls) -> str:
                return "bfloat16"

        assert issubclass(BFloat16Inner, FloatToken)
        assert issubclass(BFloat16Inner, BFloat16Token)
        assert BFloat16Inner.width_bytes == 2

        class Float32Inner(_V2FloatInner, Float32Token):
            __slots__ = ()

            def __init__(self, bits: Optional[int] = None):
                super().__init__(bits)

            @classmethod
            def _get_basename(cls) -> str:
                return "float32"

        assert issubclass(Float32Inner, FloatToken)
        assert issubclass(Float32Inner, Float32Token)
        assert Float32Inner.width_bytes == 4

        class Float64Inner(_V2FloatInner, Float64Token):
            __slots__ = ()

            def __init__(self, bits: Optional[int] = None):
                super().__init__(bits)

            @classmethod
            def _get_basename(cls) -> str:
                return "float64"

        assert issubclass(Float64Inner, FloatToken)
        assert issubclass(Float64Inner, Float64Token)
        assert Float64Inner.width_bytes == 8

        class Float80Inner(_V2FloatInner, Float80Token):
            __slots__ = ()

            def __init__(self, bits: Optional[int] = None):
                super().__init__(bits)

            @classmethod
            def _get_basename(cls) -> str:
                return "float80"

        assert issubclass(Float80Inner, FloatToken)
        assert issubclass(Float80Inner, Float80Token)
        assert Float80Inner.width_bytes == 10

        class Float128Inner(_V2FloatInner, Float128Token):
            __slots__ = ()

            def __init__(self, bits: Optional[int] = None):
                super().__init__(bits)

            @classmethod
            def _get_basename(cls) -> str:
                return "float128"

        assert issubclass(Float128Inner, FloatToken)
        assert issubclass(Float128Inner, Float128Token)
        assert Float128Inner.width_bytes == 16

        # Modifier Inner classes — parameterless category markers. Wire
        # form is exactly one type id, no payload. Single mixin captures
        # the trivial encoding; subclasses pick a basename.

        class _V2ModifierInner(TokensInner, ModifierToken, ABC):
            __slots__ = ("_type_token_id",)

            def __init__(self):
                assert vocab_manager.format_version in (1, 2), (
                    "v2 Inner classes require format_version=1 (unified) or =2 (per-binary CSV) VocabularyManager; "
                    f"got format_version={vocab_manager.format_version}"
                )
                self._type_token_id = vocab_manager._private_add_token(self._get_basename(), self.__class__)

            @classmethod
            @abstractmethod
            def _get_basename(cls) -> str:
                ...

            @classmethod
            def _from_token_ids(cls, token_ids: List[int]) -> "_V2ModifierInner":
                if len(token_ids) != 1:
                    raise ValueError(
                        f"v2 modifier token must have exactly one id, got {token_ids}"
                    )
                return cls()

            def get_token_ids(self) -> npt.NDArray[np.int_]:
                return np.array([self._type_token_id], dtype=np.int_)

            def to_string(self) -> str:
                return self._get_basename()

            def to_asm_like(self) -> str:
                return self._get_basename()

        class ThreadLocalInner(_V2ModifierInner, ThreadLocalToken):
            __slots__ = ()

            def __init__(self):
                super().__init__()

            @classmethod
            def _get_basename(cls) -> str:
                return "thread_local"

        assert issubclass(ThreadLocalInner, ModifierToken)
        assert issubclass(ThreadLocalInner, ThreadLocalToken)

        class VtableInner(_V2ModifierInner, VtableToken):
            __slots__ = ()

            def __init__(self):
                super().__init__()

            @classmethod
            def _get_basename(cls) -> str:
                return "vtable"

        assert issubclass(VtableInner, ModifierToken)
        assert issubclass(VtableInner, VtableToken)

        class CodePtrTableInner(_V2ModifierInner, CodePtrTableToken):
            __slots__ = ()

            def __init__(self):
                super().__init__()

            @classmethod
            def _get_basename(cls) -> str:
                return "code_ptr_table"

        assert issubclass(CodePtrTableInner, ModifierToken)
        assert issubclass(CodePtrTableInner, CodePtrTableToken)

        # Variant_Axis Inner — opaque-string family. Each instance holds
        # one prefixed axis string (e.g. `arch:x64`, `comp:gcc`,
        # `cver:gcc:13.2.0`, `opt:O2`, `<metakey>:<metaval>`) that
        # registers as exactly one vocab entry. Wire form: a single
        # vocab id, no inline payload. Unlike the v2 inline-digit Inner
        # classes, the "data" here IS the string — there is no payload
        # to decode separately, so `_from_token_ids([id])` resolves the
        # string back via `vocab_manager.get_token_str(id)`.
        #
        # The string itself is opaque to this class — the prefix grammar
        # is owned by `tokenizer.variant_tokens.prefixes`. Variant_Axis
        # is only the registration + round-trip surface.
        #
        # No reserved-digit-protocol restriction: variant tokens make
        # sense only in the unified vocab (format_version=1; see plan
        # memoized-booping-wren.md), but the Inner class itself does not
        # assert format_version because the token's wire form (single
        # id) is layout-independent. The unifier is the single caller
        # that registers these on a v1 unified VM.

        class VariantAxisInner(TokensInner, VariantAxisToken):
            """Represents a variant-axis identity token (opaque string)."""

            __slots__ = ("token", "_token_id")

            def __init__(self, token: str):
                if " " in token:
                    raise ValueError(f"Variant-axis token cannot contain spaces: '{token}'")
                self.token = token
                self._token_id = vocab_manager._private_add_token(token, self.__class__)

            @classmethod
            def _from_token_ids(cls, token_ids: List[int]) -> "VariantAxisInner":
                """Reconstruct a Variant_Axis token from its single-id wire slice."""
                if len(token_ids) != 1:
                    raise ValueError(
                        f"Variant-axis token must have exactly one ID, got {len(token_ids)}"
                    )
                token_str = vocab_manager.get_token_str(token_ids[0])
                if not token_str:
                    raise ValueError(
                        f"Variant-axis token id {token_ids[0]} not found in vocabulary"
                    )
                return cls(token_str)

            def get_token_ids(self) -> npt.NDArray[np.int_]:
                return np.array([self._token_id], dtype=np.int_)

            def to_string(self) -> str:
                return self.token

            def to_asm_like(self) -> str:
                return self.token

        assert issubclass(VariantAxisInner, Tokens)
        assert issubclass(VariantAxisInner, VariantAxisToken)
        assert VariantAxisInner.token_type == TokenType.VARIANT_AXIS

        class MemoryOperandTokenInner(TokensInner, MemoryOperandToken):
            """Represents memory operand symbols like [, ], +, *"""

            __slots__ = ("symbol", "_token_id")
            _token_cache = MemoryOperandToken.EnumTokenCache()

            def __init__(self, symbol: MemoryOperandSymbol):
                self.symbol = symbol
                # Register the token and cache its ID
                self._token_id = vocab_manager._private_add_token(symbol.token_str(), self.__class__)

            @classmethod
            def _from_enum(cls, symbol):
                return cls(symbol)

            @classmethod
            def _get_enum_token_cache(cls) -> MemoryOperandToken.EnumTokenCache:
                return cls._token_cache

            @classmethod
            def _from_token_ids(cls, token_ids: List[int]) -> "MemoryOperandTokenInner":
                """Reconstruct a MemoryOperandToken from token IDs"""
                if len(token_ids) != 1:
                    raise ValueError(f"Memory operand token must have exactly one ID, got {len(token_ids)}")

                token_str = vocab_manager.get_token_str(token_ids[0])
                for symbol in MemoryOperandSymbol:
                    if symbol.token_str() == token_str:
                        return cls(symbol)

                raise ValueError(f"Invalid memory operand token string: {token_str}")

            def get_token_ids(self) -> npt.NDArray[np.int_]:
                return np.array([self._token_id], dtype=np.int_)

            def to_string(self) -> str:
                return self.symbol.token_str()

            def to_asm_like(self) -> str:
                return str(self.symbol.value)

        # Ensure MemoryOperandTokenInner conforms to both protocols
        assert issubclass(MemoryOperandTokenInner, Tokens)
        assert issubclass(MemoryOperandTokenInner, MemoryOperandToken)
        assert MemoryOperandTokenInner.token_type == TokenType.MEMORY_OPERAND

        class RegisterListTokenInner(TokensInner, RegisterListToken):
            """Represents ARM register-list bracketing symbols (reglist{, }reglist, !)."""

            __slots__ = ("symbol", "_token_id")
            _token_cache = RegisterListToken.EnumTokenCache()

            def __init__(self, symbol: RegisterListSymbol):
                self.symbol = symbol
                # Register the token and cache its ID
                self._token_id = vocab_manager._private_add_token(symbol.token_str(), self.__class__)

            @classmethod
            def _from_enum(cls, symbol):
                return cls(symbol)

            @classmethod
            def _get_enum_token_cache(cls) -> RegisterListToken.EnumTokenCache:
                return cls._token_cache

            @classmethod
            def _from_token_ids(cls, token_ids: List[int]) -> "RegisterListTokenInner":
                """Reconstruct a RegisterListToken from token IDs"""
                if len(token_ids) != 1:
                    raise ValueError(f"Register list token must have exactly one ID, got {len(token_ids)}")

                token_str = vocab_manager.get_token_str(token_ids[0])
                for symbol in RegisterListSymbol:
                    if symbol.token_str() == token_str:
                        return cls(symbol)

                raise ValueError(f"Invalid register list token string: {token_str}")

            def get_token_ids(self) -> npt.NDArray[np.int_]:
                return np.array([self._token_id], dtype=np.int_)

            def to_string(self) -> str:
                return self.symbol.token_str()

            def to_asm_like(self) -> str:
                return str(self.symbol.value)

        # Ensure RegisterListTokenInner conforms to both protocols
        assert issubclass(RegisterListTokenInner, Tokens)
        assert issubclass(RegisterListTokenInner, RegisterListToken)
        assert RegisterListTokenInner.token_type == TokenType.REGISTER_LIST

        class TokenSetInner(TokensInner):
            """Represents a collection of tokens"""

            __slots__ = ("tokens",)

            def __init__(self, tokens: List[TokensInner]):
                self.tokens = tokens

            def get_token_ids(self) -> npt.NDArray[np.int_]:
                token_ids = []
                for token in self.tokens:
                    token_ids.extend(token.get_token_ids())
                return np.array(token_ids, dtype=np.int_)

            def to_string(self) -> str:
                return " ".join(token.to_string() for token in self.tokens)

            def to_asm_like(self) -> str:
                return " ".join(token.to_asm_like() for token in self.tokens)

            def __iter__(self):
                return iter(self.tokens)

            def __len__(self):
                return len(self.tokens)

            def append(self, token: TokensInner):
                self.tokens.append(token)

            def extend(self, tokens: List[TokensInner]):
                self.tokens.extend(tokens)

        # Assign the inner classes to instance variables WITHOUT the Inner suffix
        self.TokensRepl = TokensInner
        self.PlatformToken = PlatformTokenInner
        self.Valued_Const = ValuedConstTokenInner
        self.Identifier = IdentifierInner
        self.Block_Def = BlockDefInner
        self.Block = BlockInner
        self.Opaque_Const = OpaqueConstInner
        self.MemoryOperand = MemoryOperandTokenInner
        self.RegisterList = RegisterListTokenInner
        self.TokenSet = TokenSetInner
        # v2 category factories. Available on every VM but guarded at
        # construction time — instantiating any of these on a VM whose
        # format_version is outside `(1, 2)` fails the assertion in
        # `_V2IdentityInner.__init__` (and peers). v1 is the unified
        # vocab and accepts v2-shape category tokens unchanged; variant
        # tokens are additive on v1.
        self.Local_Func = LocalFuncInner
        self.Plt_Func = PltFuncInner
        self.Ext_Func = ExtFuncInner
        self.Ro_Data_Ptr = RoDataPtrInner
        self.Rw_Data_Ptr = RwDataPtrInner
        self.String_Ptr = StringPtrInner
        self.Jump_Table = JumpTableInner
        self.Block_V2 = BlockV2Inner
        self.Valued_Const_V2 = ValuedConstV2Inner
        self.Float16 = Float16Inner
        self.BFloat16 = BFloat16Inner
        self.Float32 = Float32Inner
        self.Float64 = Float64Inner
        self.Float80 = Float80Inner
        self.Float128 = Float128Inner
        self.Thread_Local = ThreadLocalInner
        self.Vtable = VtableInner
        self.Code_Ptr_Table = CodePtrTableInner
        # Variant-axis opaque-string token (v1 unified vocab only).
        # Registered on every VM so the dispatch table is complete; the
        # unifier is the only intended caller.
        self.Variant_Axis = VariantAxisInner
