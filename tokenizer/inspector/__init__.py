"""Inspector for ``batch_decode`` results — TUI tree browser.

Sibling of :mod:`tokenizer.aligned_data.loader`, not nested inside it,
because the inspector consumes the loader's public API rather than
extending it. The Textual app + tree model + render helpers land in
later phases; this package marker exists so the argparse entry +
session-lifecycle scaffold can be exercised standalone.
"""
