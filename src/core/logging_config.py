"""
Structured JSON logging configuration.

Call configure_logging() once at the start of each pipeline entrypoint so that
all logger.info/warning/error calls emit newline-delimited JSON that can be
ingested by log aggregators (Datadog, CloudWatch, etc.).

Usage:
    from src.core.logging_config import configure_logging
    configure_logging()          # INFO level, JSON to stderr
    configure_logging(level="DEBUG")
"""
import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger to emit structured JSON."""
    try:
        from pythonjsonlogger import jsonlogger

        handler = logging.StreamHandler(sys.stderr)
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
    except ImportError:
        # python-json-logger not installed — fall back to plain text so the
        # pipeline still runs (e.g. in a minimal test environment).
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )

    root = logging.getLogger()
    # Avoid adding duplicate handlers if called more than once.
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
