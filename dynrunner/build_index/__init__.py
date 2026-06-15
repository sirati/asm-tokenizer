"""dynrunner driver for the per-binary index build (phase 4).

Wraps the two existing per-binary index generators —
``tokenizer.aligned_data.realized_lengths`` (realized-length sidecars)
and ``tokenizer.aligned_data.sorted_index`` (sorted-index ``.idx``
files) — in a single :class:`~dynamic_runner.task_protocol.TaskDefinition`
so the work shards across workers via
``python -m dynrunner.build_index``. The two run as TWO task types in
ONE phase; the sorted-index task DEPENDS (per binary) on its
realized-length sibling via ``TaskInfo.task_depends_on``.
"""
