"""FtlBackend -- per-binary-CSV implementation of ``RenderBackend``.

Phase A3 deliverable per the Wave-5 plan. Consumes per-variant
``<base>_output.csv`` files via :func:`lockstep_records` +
:class:`ParsedRecord`, binds per-CSV vocab at construction, and
routes :meth:`RenderBackend.render_block` through the shared
:mod:`tokenizer.inspector._render._render_block` body.

The per-binary discovery + vocab cache lives in
:class:`tokenizer.inspector._render._ftl_backend._csv_index.CsvIndex`
(one instance per binary, factory-owned). One :class:`FtlBackend`
exists per ``FunctionNode.expand`` call -- see plan section 4 for the
Protocol contract.
"""

from ._backend import FtlBackend


__all__ = ["FtlBackend"]
