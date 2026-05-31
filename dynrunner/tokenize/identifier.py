"""Stable identifier for a tokenizer task.

Five fields name an angr/ghidra tokenization run: the binary itself, the
build platform, the compiler, the version, and the optimisation level —
plus a ``variant_id`` disambiguator for sidecar corpora where several
builds share all five (differing only on the hardening / sanitizer /
march axes the canonical-5 don't capture). The runner only needs an
opaque hashable string per work item; `identifier_key()` produces the
canonical "<binary_name>/.../<opt_level>" form (with the optional
``__<variant_id:08x>`` suffix) that Rust ingests as `Arc<str>`.
"""

from __future__ import annotations

from dataclasses import dataclass

from tokenizer.variant_info import format_variant_id_suffix


@dataclass(frozen=True)
class TokenizerIdentifier:
    binary_name: str
    platform: str
    compiler: str
    version: str
    opt_level: str
    variant_id: int = 0

    def identifier_key(self) -> str:
        """Canonical string form, suitable for use as a `RunnerIdentifier`.

        Path-style join of the canonical-5 axes, then the optional
        ``__<variant_id:08x>`` variant suffix. The suffix is REQUIRED for
        uniqueness on sidecar corpora: multiple builds of the same
        ``(binary_name, platform, compiler, version, opt_level)`` differ
        only on the variant axes (hardening / sanitizer / march), so
        without it their task ids collide and the framework rejects the
        whole task graph (``duplicate task_id … in pool``). It carries
        the SAME ``variant_id`` discriminator the per-variant output
        filename uses (see ``tokenizer.output_filename``), keeping task
        id and output file on one identity scheme. ``variant_id == 0``
        (legacy / non-sidecar) renders no suffix, so those ids are
        unchanged.
        """
        return (
            f"{self.binary_name}/{self.platform}/{self.compiler}"
            f"/{self.version}/{self.opt_level}"
            f"{format_variant_id_suffix(self.variant_id)}"
        )

    def __str__(self) -> str:
        return self.identifier_key()
