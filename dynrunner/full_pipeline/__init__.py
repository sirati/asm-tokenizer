"""Composite three-phase pipeline task.

Exposes :class:`FullPipelineTask`, a single
:class:`~dynamic_runner.task_protocol.TaskDefinition` that wraps the
existing tokenize / unify-vocab / build-memmap children so the
framework drives the entire chain on a persistent secondary mesh.
"""

from .full_pipeline_task import FullPipelineTask

__all__ = [
    "FullPipelineTask",
]
