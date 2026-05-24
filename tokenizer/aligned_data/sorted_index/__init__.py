"""Per-section depth-N length sorted-index builder + reader (skeleton).

This package will own the sorted-index builder + reader described in
``sorted-index-builder.md``. Phase 0c (this commit) lands only the
on-disk-fixture scaffolding the later phases' test suites consume; the
production modules (``builder.py`` / ``reader.py`` / ``modes.py`` /
``sampler.py``) are populated by Phase 1+.

Keeping the package skeleton empty here keeps the public API surface
strictly inside the later phases' control -- the fixtures package is
the only resident for now.
"""
