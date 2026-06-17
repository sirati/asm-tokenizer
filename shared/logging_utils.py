import logging
import sys
from pathlib import Path
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


def resilient_file_handler(
    log_file_path: Path,
    level: int = logging.INFO,
    formatter: logging.Formatter | None = None,
) -> logging.Handler:
    """Build a file log handler that NEVER raises if the log sink is unusable.

    A worker's log directory can vanish mid-run (e.g. a shared/bind-mounted
    log dir deleted on the host leaves the in-container path a *stale mount
    handle*: ``mkdir(exist_ok=True)`` then raises ``FileExistsError`` and
    ``touch``/``FileHandler`` raise ``OSError``). Logging is a side concern;
    losing the log sink must degrade to stderr, never abort the work that
    produced the logs. This helper localizes that policy so callers only ask
    for "a handler for this path" and are guaranteed a usable one.

    Returns a ``FileHandler`` at ``log_file_path`` when the directory is
    writable; otherwise a ``StreamHandler`` on stderr. The chosen handler
    carries ``formatter`` (or the project-standard format) at ``level``.
    """
    if formatter is None:
        formatter = logging.Formatter(
            "%(levelname)s | %(asctime)s,%(msecs)03d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler: logging.Handler
    try:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        log_file_path.touch()
        handler = logging.FileHandler(log_file_path, mode="a")
    except OSError as exc:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        # Emit through the fallback so the degradation is itself recorded.
        logging.getLogger(__name__).warning(
            "log file %s unusable (%s: %s); falling back to stderr",
            log_file_path,
            type(exc).__name__,
            exc,
        )
        return handler

    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


def setup_logger(
    name: str,
    level: int | None = None,
) -> tuple[logging.Logger, WarningCounterHandler]:
    """Create a logger that inherits from root logger configuration."""
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)

    counter_handler = WarningCounterHandler()
    counter_handler.attach_to_logger(logger)

    return logger, counter_handler


def setup_file_logger(
    name: str,
    log_file_path: Path,
    level: int = logging.INFO,
    console: bool = True,
    console_format: str | None = None,
) -> logging.Logger:
    """Create a logger with file handler and optional console handler.

    Args:
        name: Logger name
        log_file_path: Path to log file
        level: Logging level (default: INFO)
        console: Whether to add console handler (default: True)
        console_format: Custom format for console handler. If None, inherits from root logger.

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        "%(levelname)s | %(asctime)s,%(msecs)03d | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # If no custom format specified, inherit from root logger's handlers
        if console_format is None:
            root_logger = logging.getLogger()
            root_format = None
            root_datefmt = None

            # Find the first StreamHandler in root logger to inherit its format
            for handler in root_logger.handlers:
                if isinstance(handler, logging.StreamHandler) and handler.formatter:
                    root_format = handler.formatter._fmt
                    root_datefmt = handler.formatter.datefmt
                    break

            # Fall back to standard format if root has no stream handlers
            if root_format is None:
                root_format = "%(levelname)s | %(asctime)s,%(msecs)03d | %(message)s"
                root_datefmt = "%Y-%m-%d %H:%M:%S"

            console_formatter = logging.Formatter(root_format, datefmt=root_datefmt)
        else:
            console_formatter = logging.Formatter(console_format, datefmt="%Y-%m-%d %H:%M:%S")

        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger
