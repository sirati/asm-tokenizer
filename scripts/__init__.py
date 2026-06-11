"""Repo-level helper scripts.

Marks ``scripts`` as a package so submodule packages (e.g.
``scripts.collect_stats``) can be run via ``python -m scripts.collect_stats``
from the repo root.  Standalone single-file scripts in this directory
(run by path) are unaffected.
"""
