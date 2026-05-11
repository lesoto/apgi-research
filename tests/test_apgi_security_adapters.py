"""
Comprehensive tests for apgi_security_adapters.py module.
Aiming for 100% code coverage.
"""

import json
import os
import subprocess
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from apgi_security_adapters import (
    ConfigChecksumAdapter,
    SecurityAdapterFactory,
    SecurityContext,
    SecurityEvent,
    SecurityLevel,
    SecurityMetrics,
    SerializationSecurityAdapter,
    SubprocessSecurityAdapter,
    get_security_factory,
    set_security_factory,
)

# Store original factory for cleanup
_security_factory = get_security_factory()


class TestSecurityLevel:
    """Test SecurityLevel enum."""

    def test_security_level_values(self):
        """Test all security level values."""
        assert (
            SecurityLevel.PERMISSIVE.value == "permissive"
        )  # nosec: B101 - Test assertion
        assert (
            SecurityLevel.STANDARD.value == "standard"
        )  # nosec: B101 - Test assertion
        assert SecurityLevel.STRICT.value == "strict"  # nosec: B101 - Test assertion


class TestSecurityContext:
    """Test SecurityContext dataclass."""

    def test_default_creation(self):
        """Test default context creation."""
        context = SecurityContext()

        assert context.context_id is not None  # nosec: B101 - Test assertion
        assert context.operator_id == "anonymous"  # nosec: B101 - Test assertion
        assert context.role == "user"  # nosec: B101 - Test assertion
        assert context.subprocess_allowlist == {
            "git",
            "pytest",
            "python",
        }  # nosec: B101 - Test assertion
        assert context.pickle_allowed is False  # nosec: B101 - Test assertion
        assert context.serialization_format == "json"  # nosec: B101 - Test assertion
        assert (
            context.security_level == SecurityLevel.STANDARD
        )  # nosec: B101 - Test assertion
        assert isinstance(context.created_at, datetime)  # nosec: B101 - Test assertion

    def test_custom_creation(self):
        """Test custom context creation."""
        context = SecurityContext(
            operator_id="test_user",
            role="admin",
            subprocess_allowlist={"ls", "cat"},
            pickle_allowed=True,
            security_level=SecurityLevel.PERMISSIVE,
        )

        assert context.operator_id == "test_user"  # nosec: B101 - Test assertion
        assert context.role == "admin"  # nosec: B101 - Test assertion
        assert context.subprocess_allowlist == {
            "ls",
            "cat",
        }  # nosec: B101 - Test assertion
        assert context.pickle_allowed is True  # nosec: B101 - Test assertion
        assert (
            context.security_level == SecurityLevel.PERMISSIVE
        )  # nosec: B101 - Test assertion

    def test_post_init_with_operator(self):
        """Test post_init with provided operator_id."""
        context = SecurityContext(operator_id="specific_user")
        assert context.operator_id == "specific_user"  # nosec: B101 - Test assertion

    def test_post_init_strict_with_pickle_raises(self):
        """Test that STRICT level with pickle raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            SecurityContext(
                pickle_allowed=True,
                security_level=SecurityLevel.STRICT,
            )

        assert "Pickle not allowed in STRICT security level" in str(
            exc_info.value
        )  # nosec: B101 - Test assertion


class TestSecurityEvent:
    """Test SecurityEvent dataclass."""

    def test_default_creation(self):
        """Test default event creation."""
        event = SecurityEvent()

        assert event.event_id is not None  # nosec: B101 - Test assertion
        assert isinstance(event.timestamp, datetime)  # nosec: B101 - Test assertion
        assert event.context_id == ""  # nosec: B101 - Test assertion
        assert event.operator_id == ""  # nosec: B101 - Test assertion
        assert event.event_type == ""  # nosec: B101 - Test assertion
        assert event.resource == ""  # nosec: B101 - Test assertion
        assert event.action == ""  # nosec: B101 - Test assertion
        assert event.details == {}  # nosec: B101 - Test assertion

    def test_to_dict(self):
        """Test converting to dictionary."""
        event = SecurityEvent(
            context_id="ctx_001",
            operator_id="user_001",
            event_type="test_event",
            resource="test_resource",
            action="allowed",
            details={"key": "value"},
        )

        data = event.to_dict()

        assert data["event_id"] == event.event_id  # nosec: B101 - Test assertion
        assert (
            data["timestamp"] == event.timestamp.isoformat()
        )  # nosec: B101 - Test assertion
        assert data["context_id"] == "ctx_001"  # nosec: B101 - Test assertion
        assert data["operator_id"] == "user_001"  # nosec: B101 - Test assertion
        assert data["event_type"] == "test_event"  # nosec: B101 - Test assertion
        assert data["resource"] == "test_resource"  # nosec: B101 - Test assertion
        assert data["action"] == "allowed"  # nosec: B101 - Test assertion
        assert data["details"] == {"key": "value"}  # nosec: B101 - Test assertion


class TestSecurityMetrics:
    """Test SecurityMetrics class."""

    @pytest.fixture
    def metrics(self):
        """Create security metrics."""
        with patch("apgi_security_adapters.get_logger"):
            return SecurityMetrics()

    def test_init(self, metrics):
        """Test initialization."""
        assert metrics.allowed_operations == {}  # nosec: B101 - Test assertion
        assert metrics.denied_operations == {}  # nosec: B101 - Test assertion
        assert metrics.audit_events == []  # nosec: B101 - Test assertion

    def test_record_allowed(self, metrics):
        """Test recording allowed operation."""
        metrics.record_allowed("subprocess", "git")

        assert (
            metrics.allowed_operations["subprocess:git"] == 1
        )  # nosec: B101 - Test assertion

        # Record again
        metrics.record_allowed("subprocess", "git")
        assert (
            metrics.allowed_operations["subprocess:git"] == 2
        )  # nosec: B101 - Test assertion

    def test_record_denied(self, metrics):
        """Test recording denied operation."""
        with patch.object(metrics.logger, "warning") as mock_warning:
            metrics.record_denied("subprocess", "rm", "not in allowlist")

            assert (
                metrics.denied_operations["subprocess:rm"] == 1
            )  # nosec: B101 - Test assertion
            mock_warning.assert_called_once()

    def test_record_event(self, metrics):
        """Test recording audit event."""
        event = SecurityEvent(
            context_id="ctx_001",
            event_type="test",
        )

        metrics.record_event(event)

        assert len(metrics.audit_events) == 1  # nosec: B101 - Test assertion
        assert metrics.audit_events[0] == event  # nosec: B101 - Test assertion

    def test_record_event_pruning(self, metrics):
        """Test that events are pruned after 10000."""
        # Add 10001 events
        for i in range(10001):
            event = SecurityEvent(event_id=f"event_{i}")
            metrics.record_event(event)

        assert len(metrics.audit_events) == 10000  # nosec: B101 - Test assertion
        # Should keep the most recent
        assert (
            metrics.audit_events[0].event_id == "event_1"
        )  # nosec: B101 - Test assertion
        assert (
            metrics.audit_events[-1].event_id == "event_10000"
        )  # nosec: B101 - Test assertion

    def test_get_metrics(self, metrics):
        """Test getting metrics snapshot."""
        metrics.record_allowed("subprocess", "git")
        metrics.record_allowed("subprocess", "python")
        metrics.record_denied("subprocess", "rm", "not allowed")

        data = metrics.get_metrics()

        assert data["allowed_operations"] == {  # nosec: B101 - Test assertion
            "subprocess:git": 1,
            "subprocess:python": 1,
        }
        assert data["denied_operations"] == {
            "subprocess:rm": 1
        }  # nosec: B101 - Test assertion
        assert data["total_allowed"] == 2  # nosec: B101 - Test assertion
        assert data["total_denied"] == 1  # nosec: B101 - Test assertion
        assert data["audit_events_count"] == 0  # nosec: B101 - Test assertion


class TestSubprocessSecurityAdapter:
    """Test SubprocessSecurityAdapter class."""

    @pytest.fixture
    def adapter(self):
        """Create subprocess adapter."""
        metrics = Mock()
        with patch("apgi_security_adapters.get_logger"):
            return SubprocessSecurityAdapter(metrics)

    def test_init(self, adapter):
        """Test initialization."""
        assert adapter.metrics is not None  # nosec: B101 - Test assertion
        assert (
            adapter._original_popen == subprocess.Popen
        )  # nosec: B101 - Test assertion

    def test_create_secure_popen_allowed(self, adapter):
        """Test creating secure Popen with allowed command."""
        context = SecurityContext(subprocess_allowlist={"git"})

        with patch.object(adapter.metrics, "record_event"):
            with patch.object(adapter.metrics, "record_allowed"):
                # Use SubprocessSecurityAdapter's original Popen for mocking
                with patch.object(adapter, "_original_popen") as mock_popen:
                    secure_popen = adapter.create_secure_popen(context)
                    secure_popen(["git", "status"])

                    mock_popen.assert_called_once()
                    adapter.metrics.record_allowed.assert_called_once()

    def test_create_secure_popen_denied_strict(self, adapter):
        """Test creating secure Popen with denied command in strict mode."""
        context = SecurityContext(
            subprocess_allowlist={"git"},
            security_level=SecurityLevel.STRICT,
        )

        with patch.object(adapter.metrics, "record_event"):
            with patch.object(adapter.metrics, "record_denied"):
                secure_popen = adapter.create_secure_popen(context)

                with pytest.raises(PermissionError) as exc_info:
                    secure_popen(["rm", "-rf", "/"])

                assert "not in allowlist" in str(
                    exc_info.value
                )  # nosec: B101 - Test assertion
                adapter.metrics.record_denied.assert_called_once()

    def test_create_secure_popen_denied_permissive(self, adapter):
        """Test creating secure Popen with denied command in permissive mode."""
        context = SecurityContext(
            subprocess_allowlist={"git"},
            security_level=SecurityLevel.PERMISSIVE,
        )

        with patch.object(adapter, "_original_popen") as mock_popen:
            with patch.object(adapter, "logger"):
                with patch.object(adapter.metrics, "record_event"):
                    with patch.object(adapter.metrics, "record_denied"):
                        secure_popen = adapter.create_secure_popen(context)
                        secure_popen(["rm", "-rf", "/"])

                        mock_popen.assert_called_once()

    def test_extract_command_list(self, adapter):
        """Test extracting command from list."""
        cmd = adapter._extract_command((["git", "status"],), {})
        assert cmd == "git"  # nosec: B101 - Test assertion

    def test_extract_command_string(self, adapter):
        """Test extracting command from string."""
        cmd = adapter._extract_command(("git status",), {})
        assert cmd == "git"  # nosec: B101 - Test assertion

    def test_extract_command_kwargs(self, adapter):
        """Test extracting command from kwargs."""
        cmd = adapter._extract_command((), {"args": ["python", "script.py"]})
        assert cmd == "python"  # nosec: B101 - Test assertion

    def test_extract_command_unknown(self, adapter):
        """Test extracting unknown command."""
        cmd = adapter._extract_command((), {})
        assert cmd == "unknown"  # nosec: B101 - Test assertion

    def test_check_allowlist_match(self, adapter):
        """Test checking allowlist with matching command."""
        result = adapter._check_allowlist("/usr/bin/git", {"git"})
        assert result is True  # nosec: B101 - Test assertion

    def test_check_allowlist_no_match(self, adapter):
        """Test checking allowlist with non-matching command."""
        result = adapter._check_allowlist("/usr/bin/rm", {"git"})
        assert result is False  # nosec: B101 - Test assertion


class TestSerializationSecurityAdapter:
    """Test SerializationSecurityAdapter class."""

    @pytest.fixture
    def adapter(self):
        """Create serialization adapter."""
        metrics = Mock()
        with patch("apgi_security_adapters.get_logger"):
            return SerializationSecurityAdapter(metrics)

    def test_init(self, adapter):
        """Test initialization."""
        assert adapter.metrics is not None  # nosec: B101 - Test assertion

    def test_secure_loads_json_bytes(self, adapter):
        """Test secure loads with JSON bytes."""
        context = SecurityContext()
        test_data = json.dumps({"key": "value"}).encode("utf-8")

        with patch.object(adapter.metrics, "record_event"):
            with patch.object(adapter.metrics, "record_allowed"):
                secure_loads = adapter.create_secure_loads(context)
                result = secure_loads(test_data)

                assert result == {"key": "value"}  # nosec: B101 - Test assertion
                adapter.metrics.record_allowed.assert_called_once()

    def test_secure_loads_json_string(self, adapter):
        """Test secure loads with JSON string."""
        context = SecurityContext()
        test_data = json.dumps({"key": "value"})

        with patch.object(adapter.metrics, "record_event"):
            with patch.object(adapter.metrics, "record_allowed"):
                secure_loads = adapter.create_secure_loads(context)
                result = secure_loads(test_data)

                assert result == {"key": "value"}  # nosec: B101 - Test assertion

    def test_secure_loads_pickle_permissive(self, adapter):
        """Test secure loads with pickle in permissive mode."""
        context = SecurityContext(
            pickle_allowed=True,
            security_level=SecurityLevel.PERMISSIVE,
        )

        import pickle

        test_data = pickle.dumps({"key": "value"})

        secure_loads = adapter.create_secure_loads(context)
        result = secure_loads(test_data)

        assert result == {"key": "value"}  # nosec: B101 - Test assertion

    def test_secure_loads_pickle_denied(self, adapter):
        """Test secure loads with pickle denied."""
        context = SecurityContext(pickle_allowed=False)

        with patch.object(adapter.metrics, "record_event"):
            with patch.object(adapter.metrics, "record_denied"):
                secure_loads = adapter.create_secure_loads(context)

                with pytest.raises(ValueError) as exc_info:
                    secure_loads(b"invalid pickle data")

                assert "Pickle is disabled" in str(
                    exc_info.value
                )  # nosec: B101 - Test assertion
                adapter.metrics.record_denied.assert_called_once()


class TestConfigChecksumAdapter:
    """Test ConfigChecksumAdapter class."""

    @pytest.fixture
    def adapter(self, monkeypatch):
        """Create checksum adapter with mocked KMS key."""
        monkeypatch.setenv("APGI_KMS_KEY", "test_kms_key_12345")
        metrics = Mock()
        with patch("apgi_security_adapters.get_logger"):
            return ConfigChecksumAdapter(metrics)

    def test_init_with_env_var(self, adapter):
        """Test initialization with environment variable."""
        assert adapter.kms_key == "test_kms_key_12345"  # nosec: B101 - Test assertion

    def test_init_without_env_var(self):
        """Test initialization without environment variable."""
        metrics = Mock()

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                ConfigChecksumAdapter(metrics)

            assert "APGI_KMS_KEY environment variable must be set" in str(
                exc_info.value
            )  # nosec: B101 - Test assertion

    def test_validate_config_checksum_valid(self, adapter):
        """Test validating valid checksum."""
        context = SecurityContext()
        config = {"key1": "value1", "key2": "value2"}

        # Generate valid checksum
        import hashlib
        import hmac

        config_str = json.dumps(config, sort_keys=True)
        valid_hash = hmac.new(
            "test_kms_key_12345".encode("utf-8"),
            config_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        with patch.object(adapter.metrics, "record_event"):
            with patch.object(adapter.metrics, "record_allowed"):
                result = adapter.validate_config_checksum(config, valid_hash, context)

                assert result is True  # nosec: B101 - Test assertion
                adapter.metrics.record_allowed.assert_called_once()

    def test_validate_config_checksum_invalid(self, adapter):
        """Test validating invalid checksum."""
        context = SecurityContext()
        config = {"key1": "value1"}

        with patch.object(adapter.metrics, "record_event"):
            with patch.object(adapter.metrics, "record_denied"):
                result = adapter.validate_config_checksum(
                    config, "invalid_hash", context
                )

                assert result is False  # nosec: B101 - Test assertion
                adapter.metrics.record_denied.assert_called_once()


class TestSecurityAdapterFactory:
    """Test SecurityAdapterFactory class."""

    @pytest.fixture
    def factory(self):
        """Create security factory."""
        with patch("apgi_security_adapters.get_logger"):
            return SecurityAdapterFactory()

    def test_init(self, factory):
        """Test initialization."""
        assert factory.metrics is not None  # nosec: B101 - Test assertion
        assert factory.subprocess_adapter is not None  # nosec: B101 - Test assertion
        assert factory.serialization_adapter is not None  # nosec: B101 - Test assertion
        assert factory._checksum_adapter is None  # nosec: B101 - Test assertion

    def test_create_context_default(self, factory):
        """Test creating context with defaults."""
        context = factory.create_context()

        assert context.operator_id == "anonymous"  # nosec: B101 - Test assertion
        assert context.role == "user"  # nosec: B101 - Test assertion
        assert context.subprocess_allowlist == {
            "git",
            "pytest",
            "python",
        }  # nosec: B101 - Test assertion
        assert (
            context.security_level == SecurityLevel.STANDARD
        )  # nosec: B101 - Test assertion

    def test_create_context_custom(self, factory):
        """Test creating context with custom values."""
        context = factory.create_context(
            operator_id="test_user",
            role="admin",
            subprocess_allowlist={"ls"},
            security_level=SecurityLevel.STRICT,
        )

        assert context.operator_id == "test_user"  # nosec: B101 - Test assertion
        assert context.role == "admin"  # nosec: B101 - Test assertion
        assert context.subprocess_allowlist == {"ls"}  # nosec: B101 - Test assertion
        assert (
            context.security_level == SecurityLevel.STRICT
        )  # nosec: B101 - Test assertion

    def test_get_secure_popen(self, factory):
        """Test getting secure Popen."""
        context = SecurityContext()
        popen = factory.get_secure_popen(context)

        assert callable(popen)  # nosec: B101 - Test assertion

    def test_get_secure_loads(self, factory):
        """Test getting secure loads."""
        context = SecurityContext()
        loads = factory.get_secure_loads(context)

        assert callable(loads)  # nosec: B101 - Test assertion

    def test_checksum_adapter_lazy_init(self, factory, monkeypatch):
        """Test lazy initialization of checksum adapter."""
        monkeypatch.setenv("APGI_KMS_KEY", "test_key")

        assert factory._checksum_adapter is None  # nosec: B101 - Test assertion

        adapter = factory.checksum_adapter

        assert adapter is not None  # nosec: B101 - Test assertion
        assert factory._checksum_adapter is adapter  # nosec: B101 - Test assertion

        # Second access should return same instance
        assert factory.checksum_adapter is adapter  # nosec: B101 - Test assertion

    def test_validate_config(self, factory, monkeypatch):
        """Test validating config."""
        monkeypatch.setenv("APGI_KMS_KEY", "test_key")
        context = SecurityContext()
        config = {"test": "data"}

        with patch.object(
            factory.checksum_adapter, "validate_config_checksum"
        ) as mock_validate:
            mock_validate.return_value = True

            result = factory.validate_config(config, "hash123", context)

            assert result is True  # nosec: B101 - Test assertion
            mock_validate.assert_called_once_with(config, "hash123", context)

    def test_get_metrics(self, factory):
        """Test getting metrics."""
        metrics = factory.get_metrics()

        assert "allowed_operations" in metrics  # nosec: B101 - Test assertion
        assert "denied_operations" in metrics  # nosec: B101 - Test assertion

    def test_get_audit_events(self, factory):
        """Test getting audit events."""
        # Add some events
        factory.metrics.record_event(SecurityEvent(event_id="event_1"))
        factory.metrics.record_event(SecurityEvent(event_id="event_2"))

        events = factory.get_audit_events(limit=10)

        assert len(events) == 2  # nosec: B101 - Test assertion
        assert events[0]["event_id"] == "event_1"  # nosec: B101 - Test assertion
        assert events[1]["event_id"] == "event_2"  # nosec: B101 - Test assertion

    def test_get_audit_events_with_limit(self, factory):
        """Test getting audit events with limit."""
        # Add many events
        for i in range(150):
            factory.metrics.record_event(SecurityEvent(event_id=f"event_{i}"))

        events = factory.get_audit_events(limit=50)

        assert len(events) == 50  # nosec: B101 - Test assertion


class TestGlobalFunctions:
    """Test global functions."""

    def test_get_security_factory(self):
        """Test getting global security factory."""
        factory = get_security_factory()
        assert isinstance(
            factory, SecurityAdapterFactory
        )  # nosec: B101 - Test assertion

    def test_set_security_factory(self):
        """Test setting global security factory."""
        new_factory = SecurityAdapterFactory()
        set_security_factory(new_factory)

        assert get_security_factory() is new_factory  # nosec: B101 - Test assertion

        # Reset to original
        original_factory = get_security_factory()
        set_security_factory(original_factory)
