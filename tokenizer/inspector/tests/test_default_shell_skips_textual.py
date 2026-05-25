"""PEP 562 lazy-shim regression: ``import tokenizer.inspector._app``
must NOT pull in :mod:`textual`.

Pins cluster C-L3 H3 (plan ``inspector-followup.md`` Step 5): the
default ``nix develop`` shell does NOT ship :mod:`textual`, so the
inspector package must stay importable for non-TUI consumers (the
``__main__`` entry that prints ``--help`` without spawning the App;
test files that grep the public surface without exercising the App
itself). The PEP 562 ``__getattr__`` defers the textual import to the
moment ``InspectorApp`` / ``run_inspector`` / ``_InspectorTree`` is
actually accessed.
"""

from __future__ import annotations

import subprocess
import sys


def test_importing_inspector_app_package_does_not_load_textual() -> None:
    """``import tokenizer.inspector._app`` must complete without
    pulling any ``textual.*`` module into :data:`sys.modules`. Run in
    a subprocess so the parent test process's already-imported
    :mod:`textual` (from sibling tests that DO exercise the App)
    cannot mask the regression.
    """
    code = (
        "import sys\n"
        "import tokenizer.inspector._app  # noqa: F401\n"
        "leaked = sorted(m for m in sys.modules if m == 'textual' or m.startswith('textual.'))\n"
        "print(','.join(leaked))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
    )
    leaked = proc.stdout.strip()
    assert leaked == "", (
        f"importing tokenizer.inspector._app eagerly loaded textual modules: "
        f"{leaked!r}; the PEP 562 lazy shim must defer textual imports until "
        f"InspectorApp / run_inspector / _InspectorTree is accessed."
    )
