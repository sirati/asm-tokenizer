"""End-of-run punch-list reporter.

Single concern: dump the per-arm validated/skipped/missing counters
plus the first ten error blocks via the validator's module logger.
Lives alongside the orchestrator so the logging format stays
consistent across both arms; extracted so ``validator.py`` stops
trailing 15 lines of f-string boilerplate.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def log_validation_summary(stats) -> None:
    """Log the per-arm punch list + the first ten error blocks.

    Mirrors the previous trailing block in ``validate_memmap_output``
    byte-for-byte; callers no longer have to thread the logger or
    repeat the field-list format.
    """
    if stats.errors:
        logger.error(f"Validation found {len(stats.errors)} error(s):")
        for i, error in enumerate(stats.errors[:10], 1):
            logger.error(f"\nError {i}:\n{error}")
        if len(stats.errors) > 10:
            logger.error(f"\n... and {len(stats.errors) - 10} more errors")
    else:
        logger.info(f"Validation completed successfully!")

    logger.info(f"  Matched functions validated: {stats.matched_validated}")
    logger.info(f"  Matched functions skipped (filters): {stats.matched_skipped}")
    logger.info(f"  Matched functions in CSV only: {stats.csv_only_matched}")
    logger.info(f"  Unmatched functions validated: {stats.unmatched_validated}")
    logger.info(f"  Unmatched functions skipped (filters): {stats.unmatched_skipped}")
    logger.info(f"  Unmatched functions in CSV only: {stats.csv_only_unmatched}")
    logger.info(f"  Errors found: {len(stats.errors)}")
