"""Ring buffer logging — rotates files at max_lines, deletes oldest at max_files."""
import json
import logging
import os
from pathlib import Path
from typing import Optional

from src.utils.paths import get_logs_dir, get_log_config_path


def _load_config() -> dict:
    """Load log config from JSON or return defaults."""
    config_path = get_log_config_path()
    if config_path and config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {
        "max_lines_per_file": 1000,
        "max_files": 5,
        "log_level": "INFO",
        "log_format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
    }


class RingBufferHandler(logging.FileHandler):
    """File handler that rotates when line count exceeds max_lines."""

    def __init__(self, log_dir: Path, name: str, max_lines: int = 1000,
                 max_files: int = 5, **kwargs) -> None:
        self.log_dir = log_dir
        self.base_name = name
        self.max_lines = max_lines
        self.max_files = max_files
        self.line_count = 0
        self.file_index = self._find_next_index()

        filepath = self._current_path()
        if filepath.exists():
            self.line_count = sum(1 for _ in open(filepath, "r"))
        super().__init__(str(filepath), mode="a", **kwargs)

    def _current_path(self) -> Path:
        """Return path for current log file."""
        return self.log_dir / f"{self.base_name}_{self.file_index}.log"

    def _find_next_index(self) -> int:
        """Find the highest existing log file index."""
        existing = sorted(self.log_dir.glob(f"{self.base_name}_*.log"))
        if not existing:
            return 0
        last = existing[-1].stem
        return int(last.split("_")[-1])

    def _rotate(self) -> None:
        """Close current file, open next, delete oldest if over max_files."""
        self.close()
        self.file_index += 1
        self.line_count = 0

        # Delete oldest files if exceeding max_files
        existing = sorted(self.log_dir.glob(f"{self.base_name}_*.log"))
        while len(existing) >= self.max_files:
            oldest = existing.pop(0)
            os.remove(oldest)

        self.baseFilename = str(self._current_path())
        self.stream = self._open()

    def emit(self, record: logging.LogRecord) -> None:
        """Write a log record, rotating if needed."""
        if self.line_count >= self.max_lines:
            self._rotate()
        super().emit(record)
        self.line_count += 1


def setup_logger(name: str = "l42",
                 console: bool = True) -> logging.Logger:
    """Create and return a configured logger with ring buffer file handler.

    Args:
        name: Logger name (used for log filenames)
        console: Whether to also log to console

    Returns:
        Configured logging.Logger instance
    """
    config = _load_config()
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, config["log_level"]))
    formatter = logging.Formatter(config["log_format"],
                                  datefmt=config["date_format"])

    # Ring buffer file handler
    log_dir = get_logs_dir()
    file_handler = RingBufferHandler(
        log_dir=log_dir,
        name=name,
        max_lines=config["max_lines_per_file"],
        max_files=config["max_files"],
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
