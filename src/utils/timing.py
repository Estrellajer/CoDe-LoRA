"""
Timing and performance monitoring utilities.

Provides precise timing context managers and time statistics recording functionality.
"""

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Union


class Timer:
    """Precise timing context manager."""

    def __init__(self, name: Optional[str] = None, verbose: bool = True):
        """
        Initialize timer.

        Args:
            name: Timer name (for logging)
            verbose: Whether to print elapsed time on exit
        """
        self.name = name or "Timer"
        self.verbose = verbose
        self.start_time = None
        self.elapsed_time = None

    def __enter__(self):
        """Enter context and start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and stop timing."""
        self.elapsed_time = time.perf_counter() - self.start_time
        if self.verbose:
            print(f"{self.name}: {self.elapsed_time:.4f} seconds")
        return False

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.elapsed_time is not None:
            return self.elapsed_time
        elif self.start_time is not None:
            return time.perf_counter() - self.start_time
        else:
            return 0.0


class TimingLogger:
    """Time statistics logger."""

    def __init__(self):
        """Initialize logger."""
        self.records: List[Dict] = []

    def record(self, name: str, elapsed_time: float, metadata: Optional[Dict] = None):
        """
        Record a timing result.

        Args:
            name: Timing name
            elapsed_time: Elapsed time in seconds
            metadata: Additional metadata
        """
        record = {
            "name": name,
            "elapsed_time": elapsed_time,
            "timestamp": time.time(),
        }
        if metadata:
            record["metadata"] = metadata
        self.records.append(record)

    def get_summary(self) -> Dict:
        """
        Get time statistics summary.

        Returns:
            Dictionary containing total time, average time, and other statistics
        """
        if not self.records:
            return {}

        total_time = sum(r["elapsed_time"] for r in self.records)
        grouped = {}
        for r in self.records:
            name = r["name"]
            if name not in grouped:
                grouped[name] = {
                    "count": 0,
                    "total_time": 0.0,
                    "times": [],
                }
            grouped[name]["count"] += 1
            grouped[name]["total_time"] += r["elapsed_time"]
            grouped[name]["times"].append(r["elapsed_time"])

        summary = {
            "total_records": len(self.records),
            "total_time": total_time,
            "by_name": {
                name: {
                    "count": info["count"],
                    "total_time": info["total_time"],
                    "avg_time": info["total_time"] / info["count"],
                    "min_time": min(info["times"]),
                    "max_time": max(info["times"]),
                }
                for name, info in grouped.items()
            },
        }
        return summary

    def save(self, file_path: Union[str, Path]):
        """
        Save time records to JSON file.

        Args:
            file_path: Save path
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "records": self.records,
            "summary": self.get_summary(),
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, file_path: Union[str, Path]):
        """
        Load time records from JSON file.

        Args:
            file_path: File path
        """
        file_path = Path(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.records = data.get("records", [])


# Global time logger instance
_global_logger = TimingLogger()


def log_timing(
    name: str,
    elapsed_time: float,
    metadata: Optional[Dict] = None,
    logger: Optional[TimingLogger] = None,
):
    """
    Record a timing result (using global or specified logger).

    Args:
        name: Timing name
        elapsed_time: Elapsed time in seconds
        metadata: Additional metadata
        logger: Specified logger (defaults to global logger)
    """
    target_logger = logger or _global_logger
    target_logger.record(name, elapsed_time, metadata)


def get_timing_summary(logger: Optional[TimingLogger] = None) -> Dict:
    """
    Get time statistics summary.

    Args:
        logger: Specified logger (defaults to global logger)

    Returns:
        Time statistics summary
    """
    target_logger = logger or _global_logger
    return target_logger.get_summary()


def save_timing_log(file_path: Union[str, Path], logger: Optional[TimingLogger] = None):
    """
    Save time records to file.

    Args:
        file_path: Save path
        logger: Specified logger (defaults to global logger)
    """
    target_logger = logger or _global_logger
    target_logger.save(file_path)


def load_timing_log(file_path: Union[str, Path], logger: Optional[TimingLogger] = None):
    """
    Load time records from file.

    Args:
        file_path: File path
        logger: Specified logger (defaults to global logger)
    """
    target_logger = logger or _global_logger
    target_logger.load(file_path)

