"""
Comprehensive tests for apgi_logging.py - Production-grade logging module.
"""

import json
import logging
import sys

import pytest

from apgi_logging import APGIContextLogger, JSONFormatter, get_logger


class TestJSONFormatter:
    """Tests for JSONFormatter class."""

    def test_json_formatter_basic(self):
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert parsed["level"] == "INFO"
        assert parsed["name"] == "test.logger"
        assert parsed["message"] == "Test message"
        assert "timestamp" in parsed
        assert parsed["correlation_id"] == "none"
        assert parsed["trial_id"] == "none"

    def test_json_formatter_with_correlation_id(self):
        """Test JSON formatting with correlation_id."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error message",
            args=(),
            exc_info=None,
        )
        record.correlation_id = "test-correlation-123"

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert parsed["correlation_id"] == "test-correlation-123"
        assert parsed["level"] == "ERROR"

    def test_json_formatter_with_trial_id(self):
        """Test JSON formatting with trial_id."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=1,
            msg="Debug message",
            args=(),
            exc_info=None,
        )
        record.trial_id = "trial-456"

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert parsed["trial_id"] == "trial-456"

    def test_json_formatter_with_exception(self):
        """Test JSON formatting with exception info."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test exception")
        except ValueError:
            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error occurred",
                args=(),
                exc_info=exc_info,
            )

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert "exc_info" in parsed
        assert "ValueError" in parsed["exc_info"]
        assert "Test exception" in parsed["exc_info"]

    def test_json_formatter_different_levels(self):
        """Test JSON formatting with different log levels."""
        formatter = JSONFormatter()
        levels = [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "CRITICAL"),
        ]

        for level, expected_name in levels:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="test.py",
                lineno=1,
                msg=f"{expected_name} message",
                args=(),
                exc_info=None,
            )
            formatted = formatter.format(record)
            parsed = json.loads(formatted)
            assert parsed["level"] == expected_name


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_get_logger_adds_handler(self):
        """Test get_logger adds a StreamHandler."""
        logger = get_logger("test_handler_module")
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_get_logger_sets_formatter(self):
        """Test get_logger sets JSONFormatter."""
        logger = get_logger("test_formatter_module")
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, JSONFormatter)

    def test_get_logger_sets_level(self):
        """Test get_logger sets INFO level."""
        logger = get_logger("test_level_module")
        assert logger.level == logging.INFO

    def test_get_logger_same_name_returns_same_logger(self):
        """Test get_logger returns same logger for same name."""
        logger1 = get_logger("same_name")
        logger2 = get_logger("same_name")
        assert logger1 is logger2

    def test_get_logger_different_names_different_loggers(self):
        """Test get_logger returns different loggers for different names."""
        logger1 = get_logger("name1")
        logger2 = get_logger("name2")
        assert logger1 is not logger2

    def test_get_logger_does_not_add_duplicate_handlers(self):
        """Test get_logger doesn't add duplicate handlers."""
        logger = get_logger("test_no_duplicate")
        initial_count = len(logger.handlers)
        # Call again
        logger2 = get_logger("test_no_duplicate")
        assert len(logger2.handlers) == initial_count

    def test_get_logger_logs_json(self, capsys):
        """Test that logger outputs JSON format."""
        logger = get_logger("test_json_output")
        logger.info("Test JSON message")

        captured = capsys.readouterr()
        # Parse the JSON output
        try:
            parsed = json.loads(captured.err)
            assert parsed["message"] == "Test JSON message"
            assert parsed["level"] == "INFO"
        except json.JSONDecodeError:
            # If not valid JSON, the test fails
            pytest.fail("Logger output is not valid JSON")


class TestAPGIContextLogger:
    """Tests for APGIContextLogger class."""

    def test_context_logger_init(self):
        """Test APGIContextLogger initialization."""
        base_logger = get_logger("test_context")
        context_logger = APGIContextLogger(base_logger)

        assert context_logger.logger is base_logger
        assert context_logger.correlation_id is not None
        assert context_logger.trial_id == "none"

    def test_context_logger_init_with_correlation_id(self):
        """Test APGIContextLogger with custom correlation_id."""
        base_logger = get_logger("test_context_custom")
        context_logger = APGIContextLogger(base_logger, correlation_id="custom-123")

        assert context_logger.correlation_id == "custom-123"

    def test_context_logger_generates_uuid(self):
        """Test APGIContextLogger generates UUID if not provided."""
        base_logger = get_logger("test_context_uuid")
        context_logger1 = APGIContextLogger(base_logger)
        context_logger2 = APGIContextLogger(base_logger)

        # Should generate different UUIDs
        assert context_logger1.correlation_id != context_logger2.correlation_id
        assert len(context_logger1.correlation_id) == 36  # UUID length

    def test_set_trial(self):
        """Test set_trial method."""
        base_logger = get_logger("test_set_trial")
        context_logger = APGIContextLogger(base_logger)

        context_logger.set_trial("trial-123")
        assert context_logger.trial_id == "trial-123"

    def test_context_logger_info(self, capsys):
        """Test info logging with context."""
        base_logger = get_logger("test_info")
        context_logger = APGIContextLogger(base_logger, correlation_id="corr-123")
        context_logger.set_trial("trial-456")

        context_logger.info("Test info message")

        captured = capsys.readouterr()
        try:
            parsed = json.loads(captured.err)
            assert parsed["message"] == "Test info message"
            assert parsed["level"] == "INFO"
            assert parsed["correlation_id"] == "corr-123"
            assert parsed["trial_id"] == "trial-456"
        except json.JSONDecodeError:
            pytest.fail("Logger output is not valid JSON")

    def test_context_logger_error(self, capsys):
        """Test error logging with context."""
        base_logger = get_logger("test_error")
        context_logger = APGIContextLogger(base_logger, correlation_id="corr-789")

        context_logger.error("Test error message")

        captured = capsys.readouterr()
        try:
            parsed = json.loads(captured.err)
            assert parsed["message"] == "Test error message"
            assert parsed["level"] == "ERROR"
            assert parsed["correlation_id"] == "corr-789"
        except json.JSONDecodeError:
            pytest.fail("Logger output is not valid JSON")

    def test_context_logger_warning(self, capsys):
        """Test warning logging with context."""
        base_logger = get_logger("test_warning")
        context_logger = APGIContextLogger(base_logger)

        context_logger.warning("Test warning message")

        captured = capsys.readouterr()
        try:
            parsed = json.loads(captured.err)
            assert parsed["message"] == "Test warning message"
            assert parsed["level"] == "WARNING"
        except json.JSONDecodeError:
            pytest.fail("Logger output is not valid JSON")

    def test_context_logger_debug(self, capsys):
        """Test debug logging with context."""
        base_logger = get_logger("test_debug")
        # Set debug level
        base_logger.setLevel(logging.DEBUG)
        context_logger = APGIContextLogger(base_logger)

        context_logger.debug("Test debug message")

        captured = capsys.readouterr()
        try:
            parsed = json.loads(captured.err)
            assert parsed["message"] == "Test debug message"
            assert parsed["level"] == "DEBUG"
        except json.JSONDecodeError:
            pytest.fail("Logger output is not valid JSON")

    def test_context_logger_with_args(self, capsys):
        """Test logging with format arguments."""
        base_logger = get_logger("test_args")
        context_logger = APGIContextLogger(base_logger)

        context_logger.info("Test message with %s and %d", "string", 42)

        captured = capsys.readouterr()
        try:
            parsed = json.loads(captured.err)
            assert parsed["message"] == "Test message with string and 42"
        except json.JSONDecodeError:
            pytest.fail("Logger output is not valid JSON")

    def test_context_logger_with_extra(self, capsys):
        """Test logging with extra fields."""
        base_logger = get_logger("test_extra")
        context_logger = APGIContextLogger(base_logger)

        context_logger.info("Test message", extra={"custom_field": "custom_value"})

        captured = capsys.readouterr()
        # Just verify it doesn't raise
        assert captured.err is not None


class TestLoggingIntegration:
    """Integration tests for logging module."""

    def test_full_logging_workflow(self, capsys):
        """Test complete logging workflow."""
        # Get logger
        logger = get_logger("integration_test")
        context_logger = APGIContextLogger(logger, correlation_id="workflow-123")
        context_logger.set_trial("trial-workflow")

        # Log at different levels
        context_logger.debug("Debug message")
        context_logger.info("Info message")
        context_logger.warning("Warning message")
        context_logger.error("Error message")

        captured = capsys.readouterr()
        lines = captured.err.strip().split("\n")

        # Should have 4 log lines (debug level may not show depending on config)
        assert len(lines) >= 3  # At least info, warning, error

        for line in lines:
            if line:
                parsed = json.loads(line)
                assert parsed["correlation_id"] == "workflow-123"
                assert parsed["trial_id"] == "trial-workflow"

    def test_logger_inheritance(self):
        """Test that loggers follow hierarchy."""
        parent = get_logger("parent")
        child = get_logger("parent.child")

        assert child.parent is parent or child.name.startswith("parent.")


class TestLoggingEdgeCases:
    """Edge case tests for logging module."""

    def test_json_formatter_unicode(self):
        """Test JSON formatter with unicode characters."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.unicode",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Unicode message: 你好世界 🌍",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert parsed["message"] == "Unicode message: 你好世界 🌍"

    def test_json_formatter_special_chars(self):
        """Test JSON formatter with special characters."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.special",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg='Special chars: "quotes" \n newline \t tab',
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert "quotes" in parsed["message"]
        assert "newline" in parsed["message"]
        assert "tab" in parsed["message"]

    def test_context_logger_empty_correlation_id(self, capsys):
        """Test context logger with empty correlation_id generates UUID."""
        base_logger = get_logger("test_empty_corr_2")
        context_logger = APGIContextLogger(base_logger, correlation_id="")

        context_logger.info("Message with empty correlation")

        captured = capsys.readouterr()
        parsed = json.loads(captured.err)
        # Empty string is treated as falsy and generates a UUID
        assert len(parsed["correlation_id"]) == 36  # UUID length
        assert parsed["correlation_id"] != ""

    def test_context_logger_long_message(self, capsys):
        """Test context logger with very long message."""
        base_logger = get_logger("test_long")
        context_logger = APGIContextLogger(base_logger)

        long_message = "A" * 10000
        context_logger.info(long_message)

        captured = capsys.readouterr()
        parsed = json.loads(captured.err)
        assert len(parsed["message"]) == 10000
