"""Output-filename binary-name-slot round-trip + backward-compat.

The canonical output basename
(``<arch>-<compiler>-<compiler_version>-<opt>_<binary_name>``) is the
single seam through which the multi-binary discovery change flows: each
binary in a sidecar folder gets its OWN name in the binary-name slot.
The make-or-break correctness property is that this slot round-trips
exactly for arbitrary binary names — including dotted shared objects and
names carrying ``-`` / ``+`` / ``_`` / ``.`` — so the build_memmap
pairing walk (``match_filename`` / ``parse_binary_filename``) recovers
the same name it was written with, and never collides two binaries onto
one group.

The first four axes are ``-``-delimited and bounded by ``[^-_]``; the
``_`` before the binary name is therefore unambiguous and the greedy
``binary_name=.+`` capture (anchored by the ``_output.csv`` tail) cannot
swallow an axis. The tests below pin that with the live regex the
vocab-unifier / build_memmap pairing actually uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dynrunner.binary_selection import (
    SelectionConfig,
    compile_selection_filters,
    match_filename,
)
from tokenizer.output_filename import (
    format_output_basename,
    format_output_csv_filename,
)
from tokenizer.variant_info import VariantInfo, split_variant_id_suffix


# Arch kept to the legacy-parseable shape (no ``_`` / ``-``) so the
# binary-name slot is the only varying axis under test; distro archs
# (``x86_64``) intentionally don't parse from the filename alone — they
# round-trip via the ``_meta.json`` sidecar, covered by from_csv tests.
_ARCH = "x64"
_COMPILER = "gcc"
_VERSION = "13.2.0"
_OPT = "O2"


def _csv_filters() -> object:
    """The exact CSV-format filters the pairing walk compiles: the
    canonical format string plus the ``_output.csv`` tail, default
    permissive ``binary_name=.+``."""
    config = SelectionConfig(
        source_dir=Path("."),
        output_dir=Path("."),
        platforms=None,
        compiler=None,
        compiler_versions=None,
        opt_levels=None,
        file_format=(
            "platform-compiler-version-optimisationlevel_binaryname"
            "_out\\put.\\csv"
        ),
        version_regex=None,
        opt_regex="[oO]?([0123s])",
        name_regex=None,
        exclude_subfolders=None,
        list_files=False,
    )
    return compile_selection_filters(config)


@pytest.mark.parametrize(
    "binary_name",
    ["bc", "sqlite3", "libz.so.1.3.2", "libFLAC.so.12", "my_tool", "a-b+c.d"],
)
def test_binary_name_slot_roundtrips_through_pairing_regex(binary_name: str) -> None:
    csv_name = format_output_csv_filename(
        _ARCH, _COMPILER, _VERSION, _OPT, binary_name, variant_id=0
    )
    ident = match_filename(csv_name, _csv_filters())
    assert ident is not None, csv_name
    # The pairing walk peels any ``__<8hex>`` variant suffix; with
    # variant_id=0 there is none, so the parsed slot is the name itself.
    stripped, variant_id = split_variant_id_suffix(ident.binary_name)
    assert stripped == binary_name
    assert variant_id == 0
    assert ident.platform == _ARCH
    assert ident.compiler == _COMPILER
    assert ident.version == _VERSION
    assert ident.opt_level == _OPT


@pytest.mark.parametrize(
    "binary_name", ["bc", "sqlite3", "libz.so.1.3.2", "libFLAC.so.12"]
)
def test_binary_name_slot_roundtrips_through_from_csv(
    tmp_path: Path, binary_name: str
) -> None:
    """End-to-end: write the CSV under its canonical name and recover the
    binary-name slot via the filename parse inside ``from_csv`` (legacy
    fallback, no meta sidecar). ``pkg`` carries the binary name here
    because, absent a sidecar, the filename slot IS the recovered pkg."""
    csv_name = format_output_csv_filename(
        _ARCH, _COMPILER, _VERSION, _OPT, binary_name, variant_id=0
    )
    csv_path = tmp_path / csv_name
    csv_path.write_text("")
    info = VariantInfo.from_csv(csv_path)
    assert info.pkg == binary_name
    assert info.arch == _ARCH
    assert info.compiler == _COMPILER
    assert info.compiler_version == _VERSION
    assert info.opt == _OPT


def test_single_binary_basename_byte_identical_to_pre_change() -> None:
    """Backward-compat: when the binary-name slot equals the package
    name (every legacy / Dataset-1 / binarycorps single-binary build),
    the basename is exactly what the pre-change code emitted from
    ``variant.pkg`` — byte-for-byte. The pre-change formula was
    ``<arch>-<compiler>-<version>-<opt>_<pkg>`` (variant_id=0 → no
    suffix), reproduced here as the frozen oracle."""
    pkg = "minigzipsh"
    pre_change = f"{_ARCH}-{_COMPILER}-{_VERSION}-{_OPT}_{pkg}"
    # With binary_name == pkg, the slot value is identical.
    assert (
        format_output_basename(_ARCH, _COMPILER, _VERSION, _OPT, pkg, variant_id=0)
        == pre_change
    )
