"""Per-variant build-axis identity plus opaque pass-through metadata.

A `VariantInfo` is the canonical-4 build axes (`arch`, `compiler`,
`compiler_version`, `opt`) plus the package name (`pkg`), an integer
disambiguator (`variant_id`), and an opaque `extra_metadata` dict that
flows end-to-end without any worker inspecting it.

Two construction paths exist:

* `from_legacy_filename(path)`: 4-axis filename convention
  (`<platform>-<compiler>-<version>-<opt>_<binary>`). The pre-existing
  filename parser in `shared.binary_info` is the single source of truth
  for the regex; this module merely delegates and renames the tuple
  positions to the canonical field names. `variant_id=0`,
  `extra_metadata={}`.

* `from_sidecar(json_path)`: JSON sidecar paired with a same-stem
  directory containing the binary (filename:
  `<compiler>_<arch>_<opt>_<8hex>.json`). The 8-hex-digit filename
  suffix becomes `variant_id`. Storage-only keys (`tarball_name`,
  `variant_dir`, `drv`, `label`) are dropped at this boundary.
  Canonical fields are pulled out of the JSON; everything else (e.g.
  `flag_set`, `hardening`, `sanitizer`, `march`, ...) flows through as
  `extra_metadata` without enumeration, so new sidecar fields require
  no code changes here.

Asymmetry to be aware of: legacy filenames encode only the major
compiler version (e.g. `"10"`), whereas sidecars carry the full triple
(e.g. `"10.0.1"`). This dataclass stores whatever the source provides
verbatim — no truncation, no padding — so no information is lost from
either side. Downstream code that wants the major number must split on
`"."` itself.

Identity (`__eq__` / `__hash__`) covers the canonical-4 + `pkg` +
`variant_id`. `extra_metadata` is intentionally excluded because dicts
are unhashable and metadata is not part of variant identity (two reads
of the same variant must compare equal even if metadata acquisition
shape differs).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from shared.binary_info import (
    BinaryFilenameFormat,
    build_binary_filename_format,
    parse_binary_filename,
)

logger = logging.getLogger(__name__)

# Storage-only keys that exist in the sidecar JSON but never enter the
# pipeline-visible metadata. Listed here once so the boundary is
# auditable in one place.
_SIDECAR_STORAGE_ONLY_KEYS: frozenset[str] = frozenset(
    {"tarball_name", "variant_dir", "drv", "label"}
)

# Canonical-axis keys consumed directly from the sidecar (mapped onto
# fields of this dataclass). Kept as a constant so the
# extra_metadata derivation (everything-else) is single-sourced.
_SIDECAR_CANONICAL_KEYS: frozenset[str] = frozenset(
    {"arch", "compiler_family", "compiler_version", "optimization", "pkg"}
)

# Filename suffix carrying the variant hash, e.g.
# `clang10_armv7l-hf_Oz_15f3f338.json` -> `15f3f338`.
_SIDECAR_HASH_RE: re.Pattern = re.compile(r"_([0-9a-fA-F]{8})$")

# Canonical-format CSV / meta sidecar suffixes. Writer-side constants
# live in ``tokenizer.output_filename`` / ``run_tokenizer``; duplicated
# here to avoid pulling the disassembly providers in via import.
_OUTPUT_CSV_SUFFIX = "_output.csv"
_META_SIDECAR_SUFFIX = "_meta.json"

# Variant-id suffix on the binary_name slot: ``<binary>__<8hex>``.
# Mirrors ``dynrunner.build_memmap.memmap_builder_task._VARIANT_SUFFIX_RE``.
_VARIANT_SUFFIX_RE: re.Pattern = re.compile(
    r"^(?P<binary>.*)__(?P<hex>[0-9a-fA-F]{8})$"
)

# Canonical-axis keys present in a tokenize-worker ``_meta.json``
# (verbatim ``dataclasses.asdict(VariantInfo)``). Drives the
# filename-vs-sidecar per-axis disagreement check in ``from_csv``.
_SIDECAR_AXIS_KEYS: tuple[str, ...] = (
    "arch", "compiler", "compiler_version", "opt", "pkg", "variant_id",
)


@dataclass(frozen=True, eq=False)
class VariantInfo:
    """Canonical-4 build axes plus pkg, variant disambiguator, and
    opaque pass-through metadata. See module docstring for the
    legacy-vs-sidecar asymmetry."""

    arch: str
    compiler: str  # family, e.g. "clang"
    compiler_version: str  # legacy: "10"; sidecar: full string e.g. "10.0.1"
    opt: str
    pkg: str
    variant_id: int = 0
    extra_metadata: dict[str, Any] = field(default_factory=dict)
    # Canonical-format ``<base>`` shared by the variant's
    # ``_output.csv``, ``_meta.json``, ``_output.mapping.b64c`` (see
    # ``tokenizer.output_filename.format_output_basename``). Surfaces
    # in the per-binary ``_variants.csv`` so consumers can recover
    # filesystem identity. Default empty preserves backwards-compat
    # with pre-existing construction sites; ``VariantRegistry.write_sidecar``
    # is the codepath that requires a populated value and asserts.
    filename: str = ""

    # Cached default filename format for the legacy parser. Building it
    # is non-trivial (regex compilation), so it lives at class scope.
    _DEFAULT_LEGACY_FORMAT: ClassVar[BinaryFilenameFormat] = build_binary_filename_format()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VariantInfo):
            return NotImplemented
        return (
            self.arch == other.arch
            and self.compiler == other.compiler
            and self.compiler_version == other.compiler_version
            and self.opt == other.opt
            and self.pkg == other.pkg
            and self.variant_id == other.variant_id
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.arch,
                self.compiler,
                self.compiler_version,
                self.opt,
                self.pkg,
                self.variant_id,
            )
        )

    @classmethod
    def from_legacy_filename(
        cls,
        path: Path,
        binary_format: BinaryFilenameFormat | None = None,
    ) -> VariantInfo:
        """Build a `VariantInfo` from a 4-axis filename. Delegates the
        regex parse to `shared.binary_info.parse_binary_filename`; this
        method only renames the tuple positions and supplies defaults
        for the new sidecar-only fields."""
        fmt = binary_format if binary_format is not None else cls._DEFAULT_LEGACY_FORMAT
        parsed = parse_binary_filename(path.name, fmt)
        if parsed is None:
            raise ValueError(f"filename does not match legacy format: {path.name}")
        platform, compiler, version, opt_level, binary_name = parsed
        return cls(
            arch=platform,
            compiler=compiler,
            compiler_version=version,
            opt=opt_level,
            pkg=binary_name,
            variant_id=0,
            extra_metadata={},
        )

    @classmethod
    def from_sidecar(cls, json_path: Path) -> VariantInfo:
        """Build a `VariantInfo` from a JSON sidecar. The full
        `compiler_version` string is preserved (no truncation to major)
        — see module docstring for the rationale."""
        data: dict[str, Any] = json.loads(json_path.read_text())

        match = _SIDECAR_HASH_RE.search(json_path.stem)
        if match is None:
            raise ValueError(
                f"sidecar filename missing 8-hex variant suffix: {json_path.name}"
            )
        variant_id = int(match.group(1), 16)

        extra_metadata = {
            k: v
            for k, v in data.items()
            if k not in _SIDECAR_CANONICAL_KEYS and k not in _SIDECAR_STORAGE_ONLY_KEYS
        }

        return cls(
            arch=data["arch"],
            compiler=data["compiler_family"],
            compiler_version=data["compiler_version"],
            opt=data["optimization"],
            pkg=data["pkg"],
            variant_id=variant_id,
            extra_metadata=extra_metadata,
        )

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        meta_path: Path | None = None,
    ) -> VariantInfo:
        """Reconstruct from a ``<base>_output.csv`` + optional
        ``<base>_meta.json`` sidecar. Single entry point shared by the
        vocab-unifier discovery pass and the memmap-builder worker.

        Precedence: sidecar wins per axis (richer — full
        ``compiler_version`` vs the legacy filename's major-only); per-
        axis disagreement logs a warning. ``meta_path=None`` (default)
        tries the canonical sibling silently; an explicit ``meta_path``
        that does not exist warns (matches
        ``dynrunner/build_memmap/worker.py``'s declared-but-missing
        handling). Raises ``ValueError`` when neither source yields a
        usable identity.
        """
        base = csv_path.name.removesuffix(_OUTPUT_CSV_SUFFIX)
        caller_supplied_meta = meta_path is not None
        if meta_path is None:
            meta_path = csv_path.with_name(base + _META_SIDECAR_SUFFIX)

        sidecar_axes: dict[str, Any] | None = None
        extra_metadata: dict[str, Any] = {}
        if meta_path.exists():
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            extra_metadata = dict(raw.get("extra_metadata") or {})
            sidecar_axes = {
                k: raw[k] for k in _SIDECAR_AXIS_KEYS if k in raw
            }
        elif caller_supplied_meta:
            logger.warning(
                "meta sidecar declared but missing on disk: %s "
                "— falling back to filename-only parsing for %s",
                meta_path,
                csv_path.name,
            )

        filename_axes: dict[str, Any] = {}
        parsed = parse_binary_filename(base, cls._DEFAULT_LEGACY_FORMAT)
        if parsed is not None:
            # Greedy ``binary_name`` group swallows any optional
            # ``__<8hex>`` variant suffix; peel it back off.
            platform, compiler, version, opt_level, binary_name = parsed
            pkg, variant_id = _split_variant_id_suffix(binary_name)
            filename_axes = {
                "arch": platform, "compiler": compiler,
                "compiler_version": version, "opt": opt_level,
                "pkg": pkg, "variant_id": variant_id,
            }

        if not filename_axes and sidecar_axes is None:
            raise ValueError(
                f"cannot derive variant identity for {csv_path.name}: "
                f"filename does not match legacy format and no readable "
                f"sidecar at {meta_path}"
            )

        merged: dict[str, Any] = dict(filename_axes)
        for k, sv in (sidecar_axes or {}).items():
            fv = filename_axes.get(k)
            if fv is not None and fv != sv:
                logger.warning(
                    "variant-axis disagreement for %s on %r: "
                    "filename=%r vs sidecar=%r — sidecar wins",
                    csv_path.name, k, fv, sv,
                )
            merged[k] = sv

        return cls(
            arch=merged["arch"],
            compiler=merged["compiler"],
            compiler_version=merged["compiler_version"],
            opt=merged["opt"],
            pkg=merged["pkg"],
            variant_id=int(merged.get("variant_id", 0)),
            extra_metadata=extra_metadata,
            filename=base,
        )


def _split_variant_id_suffix(binary_name: str) -> tuple[str, int]:
    """Peel the optional ``__<8hex>`` variant suffix; mirrors
    ``dynrunner.build_memmap.memmap_builder_task._split_variant_suffix``.
    """
    match = _VARIANT_SUFFIX_RE.match(binary_name)
    if match is None:
        return binary_name, 0
    return match.group("binary"), int(match.group("hex"), 16)
