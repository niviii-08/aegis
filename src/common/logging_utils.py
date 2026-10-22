"""Structured logging formatters."""
import json
import logging

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            'level': record.levelname,
            'message': record.getMessage(),
            'name': record.name
        }
        return json.dumps(log_obj)
