#!/usr/bin/env python3
"""Generate tokenizer sidecars for bare ELF binaries by inspection.

Single concern: turn a flat directory of ELF files whose names carry NO
arch/compiler axes (e.g. the asm-dataset ``train`` split, named
``<pkg>-<src>-<binname>-<opt>-<32hex>``) into the **sidecar layout** that
``tokenizer.binary_discovery.walk_dataset`` already understands — without
adding a third discovery format.

Per emitted variant, in ``--out``:
  * ``<stem>.json``         — sidecar metadata read by ``VariantInfo.from_sidecar``
  * ``<stem>/<pkg>``        — the binary (copied, or symlinked with --link)

where ``stem`` ends in ``_<8hex>`` (the variant_id source) per
``_SIDECAR_HASH_RE``.

Axes:
  * arch              ← ELF ``e_machine`` (+ class/endian), mapped to a name
                        in ``arch_translation._ARCH_TO_PLATFORM`` (raises-on-
                        unknown there, so we only emit catalogued names).
  * compiler_family   ← ELF ``.comment`` ("clang"/"GCC")
  * compiler_version  ← ELF ``.comment`` semver numeric (e.g. "14.0.6")
  * optimization      ← filename ``-O{0..3}|Os|Oz|Ofast|Og-`` token
  * pkg               ← filename with the trailing ``-<opt>-<hash>`` stripped
                        (the binary's identity; also the on-disk binary name)
  * variant_id        ← first 8 hex of the filename's trailing hash

Compiler info is the only field that can be absent (stripped ``.comment``);
such binaries are warned-and-skipped rather than guessed.

Run inside ``nix develop`` (needs pyelftools). Example (2-binary test):
  nix develop --command python scripts/gen_sidecars.py \\
    --source ~/Downloads/train --out /tmp/train_sidecars --limit 2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

from elftools.elf.elffile import ELFFile

# Import the canonical arch table so we only ever emit arch strings the
# tokenizer accepts (it raises on unknown — fail loud, not guess).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokenizer.arch_translation import _ARCH_TO_PLATFORM  # noqa: E402

_logger = logging.getLogger("gen_sidecars")

# Filename tail: ``-<opt>-<hash>`` where opt is an O-level and hash is hex.
_TAIL_RE = re.compile(r"-(O[0-3]|Os|Oz|Ofast|Og)-([0-9a-fA-F]{8,})$")
_CLANG_RE = re.compile(r"clang version (\d+(?:\.\d+)*)")
_GCC_RE = re.compile(r"GCC:[^0-9]*(\d+(?:\.\d+)*)")


def _elf_arch(elf: ELFFile) -> str:
    """Map the ELF machine to a name present in ``_ARCH_TO_PLATFORM``.

    For families where the catalogued name encodes bitness/endianness
    (MIPS, PowerPC, RISC-V) we synthesize from ``elfclass`` +
    ``little_endian``. Raises if the result isn't catalogued."""
    machine = elf.get_machine_arch()  # 'x86','x64','ARM','AArch64','MIPS',...
    is64 = elf.elfclass == 64
    le = elf.little_endian
    simple = {"x86": "x86", "x64": "x86_64", "ARM": "arm32", "AArch64": "aarch64"}
    if machine in simple:
        arch = simple[machine]
    elif machine == "MIPS":
        arch = ("mips64el" if le else "mips64") if is64 else ("mipsel" if le else "mips")
    elif machine in ("PowerPC", "64-bit PowerPC"):
        arch = "ppc64le" if (is64 and le) else ("ppc64" if is64 else "ppc32")
    elif machine == "RISC-V":
        arch = "riscv64" if is64 else "riscv32"
    else:
        raise ValueError(f"unmapped ELF machine {machine!r}")
    if arch not in _ARCH_TO_PLATFORM:
        raise ValueError(f"derived arch {arch!r} not in _ARCH_TO_PLATFORM")
    return arch


def _elf_compiler(elf: ELFFile) -> tuple[str, str]:
    """Return (compiler_family, compiler_version) from ``.comment``.

    Raises ValueError if no recognizable compiler string is present."""
    sec = elf.get_section_by_name(".comment")
    blob = sec.data().decode("latin-1") if sec is not None else ""
    # .comment is NUL-separated; clang and GCC entries may coexist.
    m = _CLANG_RE.search(blob)
    if m is not None:
        return "clang", m.group(1)
    m = _GCC_RE.search(blob)
    if m is not None:
        return "gcc", m.group(1)
    raise ValueError("no clang/GCC version in .comment (stripped?)")


def _parse_name(name: str) -> tuple[str, str, str]:
    """(pkg, opt, hash8) from ``<pkg>-<opt>-<hash>``; opt anchored at end.

    Raises ValueError when the tail doesn't match (so non-conforming
    filenames are surfaced, not silently mis-parsed)."""
    m = _TAIL_RE.search(name)
    if m is None:
        raise ValueError(f"filename tail not '-<opt>-<hash>': {name}")
    opt = m.group(1)
    hash8 = m.group(2)[:8]
    pkg = name[: m.start()]
    if not pkg:
        raise ValueError(f"empty pkg after stripping tail: {name}")
    return pkg, opt, hash8


def generate_one(binary: Path, out_dir: Path, link: bool, source_label: str) -> bool:
    """Emit one sidecar + binary into ``out_dir``. Returns True on success.

    Idempotent: skips if ``<stem>.json`` already exists."""
    try:
        pkg, opt, hash8 = _parse_name(binary.name)
        with open(binary, "rb") as fh:
            elf = ELFFile(fh)
            arch = _elf_arch(elf)
            comp_family, comp_version = _elf_compiler(elf)
    except (ValueError, Exception) as exc:  # noqa: BLE001 — surface + skip
        _logger.warning("skip %s: %s", binary.name, exc)
        return False

    stem = f"{pkg}-{opt}_{hash8}"
    json_path = out_dir / f"{stem}.json"
    variant_dir = out_dir / stem
    binary_path = variant_dir / pkg

    if json_path.exists() and binary_path.exists():
        return True  # idempotent skip

    variant_dir.mkdir(parents=True, exist_ok=True)
    if binary_path.exists() or binary_path.is_symlink():
        binary_path.unlink()
    if link:
        binary_path.symlink_to(binary.resolve())
    else:
        shutil.copy2(binary, binary_path)

    sidecar = {
        "arch": arch,
        "compiler_family": comp_family,
        "compiler": f"{comp_family}{comp_version.split('.')[0]}",
        "compiler_version": comp_version,
        "optimization": opt,
        "pkg": pkg,
        "source": source_label,
    }
    json_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    _logger.info(
        "%s -> %s (arch=%s %s %s %s)", binary.name, stem, arch, comp_family, comp_version, opt
    )
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path, help="flat dir of bare ELFs")
    ap.add_argument("--out", required=True, type=Path, help="sidecar staging dir")
    ap.add_argument("--limit", type=int, default=0, help="max binaries (0 = all)")
    ap.add_argument("--name", default="", help="only files whose name contains this substring")
    ap.add_argument("--link", action="store_true", help="symlink binary instead of copy (local only)")
    ap.add_argument(
        "--source-label",
        default="",
        help="value for the sidecar 'source' field (default: basename of --source)",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    source_label = args.source_label or args.source.resolve().name

    args.out.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    with os.scandir(args.source) as it:
        for entry in it:
            if not entry.is_file(follow_symlinks=False):
                continue
            if args.name and args.name not in entry.name:
                continue
            if generate_one(Path(entry.path), args.out, args.link, source_label):
                done += 1
                if args.limit and done >= args.limit:
                    break
            else:
                skipped += 1
    _logger.info("generated %d sidecar(s), skipped %d, out=%s", done, skipped, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
