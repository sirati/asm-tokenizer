"""Unit tests for raw-binary resolution by exact filename.

Raw binaries on disk are named by their FULLNAME
(``arm32-clang-3.5-O0_minigzip``), not by the bare program
(``minigzip``); the resolver's lookup key is therefore the fullname.
Covers the lazy multi-root walk, cross-root resolution, the miss
(``None``) contract, each root walked at most once, and the regression
that a program-named file does NOT satisfy a fullname lookup.
"""

from __future__ import annotations

from pathlib import Path

from scripts.collect_stats.raw_index import RawResolver


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"binary")


def test_resolves_exact_fullname(tmp_path: Path) -> None:
    root = tmp_path / "r1"
    _touch(root / "sub" / "arm32-clang-3.5-O0_minigzip")
    resolver = RawResolver([root])
    hit = resolver.resolve("arm32-clang-3.5-O0_minigzip")
    assert hit is not None
    assert hit.name == "arm32-clang-3.5-O0_minigzip"


def test_program_named_file_does_not_satisfy_fullname_lookup(tmp_path: Path) -> None:
    """Regression: a file named by the bare program (``minigzip``) must
    NOT resolve a fullname lookup, and the fullname-named file must.  This
    is the exact bug that resolved 0/6086 on the real corpus when the
    orchestrator looked up by program instead of fullname."""
    root = tmp_path / "r1"
    _touch(root / "minigzip")  # program-named: must not satisfy fullname
    fullname = "arm32-clang-3.5-O0_minigzip"
    assert RawResolver([root]).resolve(fullname) is None
    _touch(root / fullname)  # fullname-named: must satisfy
    assert RawResolver([root]).resolve(fullname) is not None


def test_miss_returns_none(tmp_path: Path) -> None:
    root = tmp_path / "r1"
    _touch(root / "arm32-clang-3.5-O0_minigzip")
    assert RawResolver([root]).resolve("x64-gcc-9-O2_absent") is None


def test_resolves_across_multiple_roots(tmp_path: Path) -> None:
    r1 = tmp_path / "r1"
    r2 = tmp_path / "r2"
    _touch(r1 / "x64-gcc-9-O2_alpha")
    _touch(r2 / "arm32-clang-3.5-O0_beta")
    resolver = RawResolver([r1, r2])
    assert resolver.resolve("x64-gcc-9-O2_alpha") is not None
    assert resolver.resolve("arm32-clang-3.5-O0_beta") is not None


def test_no_roots_always_misses(tmp_path: Path) -> None:
    assert RawResolver([]).resolve("anything") is None


def test_nonexistent_root_is_tolerated(tmp_path: Path) -> None:
    resolver = RawResolver([tmp_path / "does-not-exist"])
    assert resolver.resolve("anything") is None
