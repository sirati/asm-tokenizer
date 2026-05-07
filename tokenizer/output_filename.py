"""Single source of truth for the per-variant output filename.

Both the task side (``dynrunner.tokenize.tokenizer_task.get_output_filename_pattern``,
which the framework's skip-existing pass calls *pre*-dispatch to decide
which items to drop) and the worker side
(``tokenizer.run_tokenizer.run_tokenizer``, which writes the CSV / meta
sidecar *post*-dispatch) need to compose the same filename for a given
variant. If the two sides disagree the skip-existing fast-path treats
already-completed work as new and re-runs it (best case) or worse — a
later phase pairs CSV ↔ meta by the canonical-format filename in
``dynrunner.build_memmap.memmap_builder_task``, so a mismatch breaks
pairing.

The filename is built from the canonical-4 build axes plus the package
name (the same five fields the legacy filename convention encodes,
preserved as identity on ``VariantInfo``). Sidecar variants append a
``__<variant_id:08x>`` suffix to the binary-name slot so multiple builds
of the same canonical-4 + pkg disambiguate; ``variant_id == 0`` (the
legacy default) emits the bare canonical form so legacy outputs stay
byte-identical.

Current dataset invariant: each sidecar tarball contains exactly one
binary, named after ``pkg`` (e.g. ``pkg=hello`` → ``./hello`` inside
the archive). This module uses ``pkg`` as the binary-name placeholder
in the canonical-format filename, which is correct under that
invariant. When the dataset gains multi-binary tarballs the discovery
side will need to emit one TaskInfo per (tarball, binary) and pass the
chosen binary's actual filename in here — at that point the parameter
name ``pkg`` here becomes ``binary_name`` and the call sites pass the
extracted member's name instead. The fix is single-call-site and the
change is mechanical because this helper is the single seam.
"""

from __future__ import annotations


_OUTPUT_CSV_SUFFIX = "_output.csv"


def format_output_basename(
    arch: str,
    compiler: str,
    compiler_version: str,
    opt: str,
    pkg: str,
    variant_id: int,
) -> str:
    """Return ``<base>``: the prefix shared by ``<base>_output.csv``,
    ``<base>_meta.json``, and ``<base>_output.mapping.b64c``.

    Composes the canonical-4-axis legacy filename convention
    (``<arch>-<compiler>-<compiler_version>-<opt>_<pkg>``) so the
    ``parse_binary_filename`` regex in ``shared.binary_info`` (which
    drives ``dynrunner.build_memmap.memmap_builder_task``'s pairing
    walk) matches both legacy and sidecar outputs uniformly.

    For sidecar variants (``variant_id != 0``) the binary-name slot
    grows a ``__<variant_id:08x>`` suffix; the pairing walk peels that
    off via ``_VARIANT_SUFFIX_RE`` to reduce the tuple key to the
    canonical-5 + ``variant_id`` shape expected by
    ``_split_variant_suffix``.

    For legacy variants (``variant_id == 0``) the suffix is omitted so
    the emitted filename equals the input binary's filename byte-for-
    byte (legacy output paths stay invariant).
    """
    base = f"{arch}-{compiler}-{compiler_version}-{opt}_{pkg}"
    if variant_id == 0:
        return base
    return f"{base}__{variant_id:08x}"


def format_output_csv_filename(
    arch: str,
    compiler: str,
    compiler_version: str,
    opt: str,
    pkg: str,
    variant_id: int,
) -> str:
    """Return the per-variant output CSV filename
    (``<base>_output.csv``). Convenience wrapper around
    ``format_output_basename`` that appends the suffix the tokenize
    phase emits and the build_memmap phase looks for.
    """
    return f"{format_output_basename(arch, compiler, compiler_version, opt, pkg, variant_id)}{_OUTPUT_CSV_SUFFIX}"
