"""Deterministic + collision-free rename for Ghidra placeholder names.

Single concern: turn a Ghidra-emitted placeholder function name (any
``Symbol`` whose ``getSource() == SourceType.DEFAULT`` — typically
``FUN_<hex>``, ``LAB_<hex>``, ``thunk_FUN_<hex>``, etc.) into a stable
opaque label that

1. Is identical across re-runs of the SAME binary
   (deterministic given the same Ghidra analysis state — the raw name
   and the per-binary identity-hash are inputs);
2. Is essentially never equal to a placeholder generated for the
   SAME raw name in a DIFFERENT binary
   (the per-binary identity-hash XOR-shifts the address-derived
   portion of the placeholder into a binary-specific namespace);
3. Cannot collide with any real ELF/PE/C/C++ symbol
   (the literal SPACE between ``unnamed`` and ``@`` is disallowed in
   every linker / language symbol grammar in our pipeline);
4. Is filesystem-safe + URL-safe (URL-safe base64 alphabet, no
   trailing ``=`` padding).

The non-placeholder branch is the identity function: real symbol
names flow through unchanged. The provider precomputes the binary
identity hash ONCE at startup (the per-binary half is a constant),
so the per-function hot path pays one 16-byte blake2b digest plus
one bytewise XOR + one urlsafe-b64encode.

Why not feed this through ``canonical_function_name`` instead?
``canonical_function_name`` is a pure function over
``(name, comment, identity_key)`` that produces the on-disk
cross-ISA-stable label. The placeholder-detection concern is
UPSTREAM of that contract: by the time canonical_function_name sees
a name, the placeholder has already been replaced with a stable
opaque label. Downstream consumers stay completely unaware that the
input was ever a Ghidra placeholder.

Hash choice: ``hashlib.blake2b(digest_size=16)`` (128-bit). This is a
uniqueness primitive, not a cryptographic primitive: the XOR with the
per-binary hash guarantees that distinct binaries never share the
same hashed-name namespace, which is the actual property the user
specified.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = (
    "PLACEHOLDER_PREFIX",
    "compute_binary_identity_hash",
    "placeholder_renamed_name",
)


# Literal space is intentional: every linker / language symbol grammar
# in our pipeline (ELF, PE, C, C++ mangled / demangled, Rust) disallows
# spaces in symbol names. The space therefore makes structural collision
# with a real symbol impossible.
PLACEHOLDER_PREFIX = "unnamed @"

# 128-bit digest. Sufficient for our uniqueness requirement; matches
# the user-specified XOR-of-two-128-bit-hashes scheme.
_DIGEST_SIZE = 16

# String returned by ``str(SourceType.DEFAULT)`` at the JPype boundary
# (JPype renders Java enum values as their constant name). Using the
# string comparison keeps the helper unit-testable without booting the
# JVM and shields the call site from JPype's evolving enum-import
# idioms.
_DEFAULT_SOURCE_STR = "DEFAULT"


def _hash128(payload: bytes) -> bytes:
    """Return a 16-byte blake2b digest of ``payload``."""
    return hashlib.blake2b(payload, digest_size=_DIGEST_SIZE).digest()


def _sidecar_json_path(binary_path: Path) -> Path:
    """Return the dataset's per-variant sidecar JSON path adjacent to
    ``binary_path``, if it exists.

    Layout convention (see :mod:`tokenizer.binary_discovery`): a
    sidecar-format variant lives at ``<dir>/<stem>/<pkg>`` with metadata
    at ``<dir>/<stem>.json``. Legacy 4-axis filenames have no sidecar
    JSON adjacent to them; the path returned for those still resolves
    (to a file that does not exist) and the caller handles the
    ``not is_file()`` branch.
    """
    return binary_path.parent.parent / f"{binary_path.parent.name}.json"


def compute_binary_identity_hash(binary_path: Path) -> bytes:
    """Return the 16-byte per-binary identity hash.

    Folded inputs (always present):

    * The binary file's absolute path as UTF-8 bytes.

    Folded inputs (when available):

    * The raw content of the dataset's per-variant sidecar JSON (see
      :func:`_sidecar_json_path` for the discovery layout). For legacy
      4-axis dataset entries that have no sidecar JSON, this input is
      omitted and the binary-path hash stands alone — still distinct
      across binaries because absolute paths are unique within a
      tokenize run.

    XOR-combining the two 128-bit hashes preserves the uniform
    distribution of each input (XOR is a bijection on fixed-width
    bytes) and matches the user-specified scheme verbatim.

    Called ONCE per provider instance; the result is stashed on the
    provider and threaded into each function view for the cheap
    per-function rename step.
    """
    path_hash = _hash128(str(binary_path.absolute()).encode("utf-8"))
    sidecar = _sidecar_json_path(binary_path)
    if not sidecar.is_file():
        return path_hash
    try:
        sidecar_bytes = sidecar.read_bytes()
    except OSError:
        # Defensive: a sidecar that exists but is unreadable degrades
        # to the path-only hash rather than crashing provider
        # construction.
        return path_hash
    # Canonicalise the JSON content before hashing so cosmetic
    # whitespace / key-order differences across re-dumps don't shift
    # the digest. Falling back to the raw bytes (preserving
    # determinism on a per-file basis) when the content isn't valid
    # JSON.
    try:
        canonical = json.dumps(
            json.loads(sidecar_bytes), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (ValueError, UnicodeDecodeError):
        canonical = sidecar_bytes
    sidecar_hash = _hash128(canonical)
    return bytes(a ^ b for a, b in zip(path_hash, sidecar_hash))


def placeholder_renamed_name(
    raw_name: str,
    source: Any,
    binary_id_hash: bytes,
) -> str:
    """Return the deterministic + collision-free rename for a Ghidra
    placeholder function, or ``raw_name`` unchanged for any real
    symbol.

    The branch predicate is ``str(source) == "DEFAULT"`` — the
    JPype-rendered name of Ghidra's
    ``SourceType.DEFAULT`` enum constant. ``SourceType.IMPORTED`` /
    ``USER_DEFINED`` / ``ANALYSIS`` (every name that Ghidra recovered
    from a real symbol table, demangler hit, or analysis pass) flow
    through unchanged.

    The placeholder branch builds the name from two parts:

    * ``hash128(raw_name.encode("utf-8"))`` — the per-function part.
      Ghidra's placeholders embed the function's entry address in
      their text (``FUN_00010000``), so the raw-name digest is
      uniformly distributed across all placeholders in the same
      binary.
    * The per-binary identity hash (precomputed by the provider).
      Bytewise XOR mixes the two so that the same raw placeholder
      ``FUN_00010000`` in two different binaries produces two
      different combined digests.

    The combined 128-bit digest is rendered as URL-safe base64 with
    ``=`` padding stripped → 22 ASCII chars in ``[A-Za-z0-9_-]``.
    The literal SPACE in ``unnamed @`` guarantees structural
    non-collision with any real symbol.
    """
    if str(source) != _DEFAULT_SOURCE_STR:
        return raw_name
    name_hash = _hash128(raw_name.encode("utf-8"))
    combined = bytes(a ^ b for a, b in zip(name_hash, binary_id_hash))
    label = base64.urlsafe_b64encode(combined).rstrip(b"=").decode("ascii")
    return f"{PLACEHOLDER_PREFIX}{label}"
