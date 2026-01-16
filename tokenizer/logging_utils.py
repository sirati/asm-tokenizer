import logging
import sys
from typing import Optional


class WarningCounterHandler(logging.Handler):
    def __init__(self, level: int = logging.WARNING):
        super().__init__(level)
        self.warning_count = 0
        self.error_count = 0
        self._loggers: list[logging.Logger] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.ERROR:
            self.error_count += 1
        elif record.levelno >= logging.WARNING:
            self.warning_count += 1

    def reset(self) -> None:
        self.warning_count = 0
        self.error_count = 0

    def get_counts(self) -> tuple[int, int]:
        return self.warning_count, self.error_count

    def attach_to_logger(self, logger: logging.Logger) -> None:
        logger.addHandler(self)
        if logger not in self._loggers:
            self._loggers.append(logger)

    def unregister(self) -> None:
        for logger in self._loggers:
            logger.removeHandler(self)
        self._loggers.clear()


def setup_logger(
    name: str,
    level: int | None = None,
) -> tuple[logging.Logger, WarningCounterHandler]:
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)

    counter_handler = WarningCounterHandler()
    counter_handler.attach_to_logger(logger)

    return logger, counter_handler
