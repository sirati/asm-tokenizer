"""dynrunner driver for the per-binary-group memmap builder.

The standalone CLI in `tokenizer/memmap_builder/__main__.py` runs the
whole pipeline single-process. This package wraps the same library
function (`tokenizer.memmap_builder.builder.build_memmap_files`) in a
`dynamic_runner.task_protocol.TaskDefinition` so the work can be sharded
across workers via `python -m dynrunner.build_memmap`.
"""
