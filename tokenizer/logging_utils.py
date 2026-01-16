import logging
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


def remove_stream_handlers(logger: logging.Logger) -> None:
    """Remove all stream handlers (stdout/stderr) from logger to avoid blocking."""
    handlers_to_remove = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    for handler in handlers_to_remove:
        logger.removeHandler(handler)


def setup_logger(
    name: str,
    level: int | None = None,
    log_file: Optional[str] = None,
    remove_streams: bool = False,
) -> tuple[logging.Logger, WarningCounterHandler]:
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)

    if remove_streams:
        remove_stream_handlers(logger)
        root_logger = logging.getLogger()
        remove_stream_handlers(root_logger)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(levelname)s | %(asctime)s,%(msecs)03d | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    counter_handler = WarningCounterHandler()
    counter_handler.attach_to_logger(logger)

    return logger, counter_handler
