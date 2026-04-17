"""
Production-grade logging for the APGI system.
"""

import logging
import json
import uuid
from typing import Optional


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "none"),
            "trial_id": getattr(record, "trial_id", "none"),
        }
        if record.exc_info:
            log_data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = JSONFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class APGIContextLogger:
    def __init__(self, logger: logging.Logger, correlation_id: Optional[str] = None):
        self.logger = logger
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.trial_id = "none"

    def set_trial(self, trial_id: str):
        self.trial_id = trial_id

    def _log(self, level: int, msg: str, *args, **kwargs):
        extra = kwargs.get("extra", {})
        extra["correlation_id"] = self.correlation_id
        extra["trial_id"] = self.trial_id
        kwargs["extra"] = extra
        self.logger.log(level, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._log(logging.INFO, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._log(logging.ERROR, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._log(logging.WARNING, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._log(logging.DEBUG, msg, *args, **kwargs)
