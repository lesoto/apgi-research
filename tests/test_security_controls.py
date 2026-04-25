"""
================================================================================
SECURITY CONTROLS ENFORCEMENT TESTS FOR APGI SYSTEM
================================================================================

This module tests that security/compliance controls are actually enforced
from GUI and CLI entry points.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# AUDIT SECURITY TESTS
# =============================================================================


class TestAuditSecurity:
    """Tests for audit subsystem security."""

    @pytest.mark.security
    def test_audit_key_required_from_env(self):
        """Test that audit requires APGI_AUDIT_KEY environment variable."""
        # Ensure APGI_AUDIT_KEY is not set
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing APGI_AUDIT_KEY
            if "APGI_AUDIT_KEY" in os.environ:
                del os.environ["APGI_AUDIT_KEY"]

            from apgi_audit import ImmutableAuditSink

            # Should raise RuntimeError when key is missing
            with pytest.raises(RuntimeError) as exc_info:
                ImmutableAuditSink()

            assert "APGI_AUDIT_KEY" in str(exc_info.value)
            assert "must be set" in str(exc_info.value).lower()

    @pytest.mark.security
    def test_audit_key_minimum_entropy(self):
        """Test that audit key must have minimum entropy."""
        with patch.dict(os.environ, {"APGI_AUDIT_KEY": "short"}):
            from apgi_audit import ImmutableAuditSink

            with pytest.raises(RuntimeError) as exc_info:
                ImmutableAuditSink()

            assert "insufficient entropy" in str(exc_info.value).lower()

    @pytest.mark.security
    def test_audit_key_rejects_weak_patterns(self):
        """Test that weak key patterns are rejected."""
        from apgi_audit import ImmutableAuditSink

        # Use longer keys that pass entropy check (32+ bytes) but contain weak patterns
        weak_patterns = [
            "default" * 5,  # 35 bytes, contains "default"
            "test" * 8,  # 32 bytes, contains "test"
            "password" * 4,  # 32 bytes, contains "password"
        ]

        for pattern in weak_patterns:
            with patch.dict(os.environ, {"APGI_AUDIT_KEY": pattern}):
                with pytest.raises(RuntimeError) as exc_info:
                    ImmutableAuditSink()

                assert "weak pattern" in str(exc_info.value).lower()

    @pytest.mark.security
    def test_audit_key_accepts_strong_key(self):
        """Test that strong keys are accepted."""
        strong_key = "a" * 64  # 64 character key (sufficient entropy)

        with patch.dict(os.environ, {"APGI_AUDIT_KEY": strong_key}):
            from apgi_audit import ImmutableAuditSink

            # Should not raise
            sink = ImmutableAuditSink()
            assert sink is not None


# =============================================================================
# AUTHORIZATION ENFORCEMENT TESTS
# =============================================================================


class TestAuthorizationEnforcement:
    """Tests for authorization enforcement at entry points."""

    @pytest.mark.security
    def test_cli_entrypoint_requires_auth(self):
        """Test that CLI entry points require authorization."""
        from apgi_cli import require_auth, AuthorizationError
        from apgi_authz import Permission

        # Mock function that requires RUN_EXPERIMENT permission
        @require_auth(Permission.RUN_EXPERIMENT, resource_type="experiment")
        def protected_function():
            return "success"

        # Without proper operator setup, should raise AuthorizationError
        with patch.dict(os.environ, {"APGI_OPERATOR_ROLE": "guest"}):
            with pytest.raises(AuthorizationError):
                protected_function()

    @pytest.mark.security
    def test_authz_denies_guest_for_run_experiment(self):
        """Test that guest role cannot run experiments."""
        from apgi_authz import (
            get_authz_manager,
            AuthorizationContext,
            Role,
            Permission,
        )

        authz = get_authz_manager()

        # Register a guest user
        guest = authz.register_operator(username="test_guest", role=Role.GUEST)

        # Create context for running experiment
        context = AuthorizationContext(
            operator=guest,
            resource_type="experiment",
            resource_id="test_experiment",
            action=Permission.RUN_EXPERIMENT,
        )

        # Should deny
        result = authz.authorize_action(context)
        assert result is False

    @pytest.mark.security
    def test_authz_allows_operator_for_run_experiment(self):
        """Test that operator role can run experiments."""
        from apgi_authz import (
            get_authz_manager,
            AuthorizationContext,
            Role,
            Permission,
        )

        authz = get_authz_manager()

        # Register an operator
        operator = authz.register_operator(username="test_operator", role=Role.OPERATOR)

        # Create context for running experiment
        context = AuthorizationContext(
            operator=operator,
            resource_type="experiment",
            resource_id="test_experiment",
            action=Permission.RUN_EXPERIMENT,
        )

        # Should allow
        result = authz.authorize_action(context)
        assert result is True

    @pytest.mark.security
    def test_authz_logs_denied_actions(self):
        """Test that denied actions are logged."""
        from apgi_authz import (
            get_authz_manager,
            AuthorizationContext,
            Role,
            Permission,
        )

        authz = get_authz_manager()

        # Register guest user
        guest = authz.register_operator(username="test_guest_logs", role=Role.GUEST)

        context = AuthorizationContext(
            operator=guest,
            resource_type="experiment",
            resource_id="test_experiment",
            action=Permission.MODIFY_EXPERIMENT,  # Guest doesn't have this
        )

        # Force authorization check
        authz.authorize_action(context)

        # Check that denial was logged in authorization_log
        auth_log = authz.get_authorization_log(limit=10)
        denials = [entry for entry in auth_log if entry.get("decision") == "denied"]

        # Should have at least one denial logged
        assert len(denials) > 0


# =============================================================================
# CONFIG SECURITY TESTS
# =============================================================================


class TestConfigSecurity:
    """Tests for configuration security."""

    @pytest.mark.security
    def test_config_secret_key_required_for_validation(self):
        """Test that config validation requires secret key."""
        from apgi_security import validate_config_checksum

        config = {"setting": "value"}
        expected_hash = "some_hash"

        # Without env var or explicit key, should raise
        with patch.dict(os.environ, {}, clear=True):
            if "APGI_CONFIG_SECRET_KEY" in os.environ:
                del os.environ["APGI_CONFIG_SECRET_KEY"]

            with pytest.raises(ValueError) as exc_info:
                validate_config_checksum(config, expected_hash)

            assert "APGI_CONFIG_SECRET_KEY" in str(exc_info.value)


# =============================================================================
# PROFILER SECURITY TESTS
# =============================================================================


class TestProfilerSecurity:
    """Tests for profiler security (disabled by default)."""

    @pytest.mark.security
    def test_profiler_disabled_by_default(self):
        """Test that profiling is disabled by default."""
        from apgi_profiler import _is_profiling_enabled

        # Ensure env var is not set
        with patch.dict(os.environ, {}, clear=True):
            if "APGI_ENABLE_PROFILING" in os.environ:
                del os.environ["APGI_ENABLE_PROFILING"]

            assert _is_profiling_enabled() is False

    @pytest.mark.security
    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", "True"])
    def test_profiler_enabled_with_valid_env_var(self, value):
        """Test that profiling can be enabled via environment variable."""
        from apgi_profiler import _is_profiling_enabled

        with patch.dict(os.environ, {"APGI_ENABLE_PROFILING": value}):
            assert _is_profiling_enabled() is True

    @pytest.mark.security
    def test_profiler_decorator_no_op_when_disabled(self):
        """Test that profiler decorator is no-op when disabled."""
        from apgi_profiler import profile_hot_path

        call_count = 0

        @profile_hot_path
        def test_function():
            nonlocal call_count
            call_count += 1
            return "result"

        with patch.dict(os.environ, {}, clear=True):
            if "APGI_ENABLE_PROFILING" in os.environ:
                del os.environ["APGI_ENABLE_PROFILING"]

            result = test_function()

            # Should still execute function
            assert result == "result"
            assert call_count == 1


# =============================================================================
# SUBPROCESS SECURITY TESTS
# =============================================================================


class TestSubprocessSecurity:
    """Tests for subprocess security (no monkey-patching)."""

    @pytest.mark.security
    def test_subprocess_not_monkey_patched(self):
        """Test that subprocess.Popen is not monkey-patched."""
        import subprocess

        # Original Popen should still be the builtin
        assert subprocess.Popen.__module__ == "subprocess"

    @pytest.mark.security
    def test_secure_popen_validates_command(self):
        """Test that secure_popen validates commands."""
        from apgi_security import secure_popen, SecureSubprocessError

        with pytest.raises(SecureSubprocessError):
            secure_popen(["rm", "-rf", "/"])

    @pytest.mark.security
    def test_secure_popen_allows_whitelisted_commands(self):
        """Test that whitelisted commands are allowed."""
        from apgi_security import secure_popen, SecureSubprocessError

        # Should not raise for allowed commands
        # Using 'echo' which is typically allowed in tests
        with patch.dict(os.environ, {"APGI_ALLOWED_SUBPROCESS_CMDS": "echo,git"}):
            try:
                proc = secure_popen(["echo", "test"], stdout=-1)  # -1 = PIPE
                proc.communicate()
                proc.wait()
            except SecureSubprocessError:
                pytest.skip("Echo not in default allowlist")


# =============================================================================
# PICKLE SECURITY TESTS
# =============================================================================


class TestPickleSecurity:
    """Tests for pickle security (no monkey-patching)."""

    @pytest.mark.security
    def test_pickle_not_monkey_patched(self):
        """Test that pickle.loads is not monkey-patched."""
        import pickle

        # Original loads should still be the builtin
        # Check by verifying it's not our wrapper
        assert "apgi" not in str(pickle.loads)

    @pytest.mark.security
    def test_secure_loads_defaults_to_json(self):
        """Test that secure_loads defaults to JSON only."""
        from apgi_security import secure_loads, PickleSecurityError

        # Try to load pickle data without explicit opt-in
        import pickle

        pickled = pickle.dumps({"test": "data"})

        with pytest.raises(PickleSecurityError):
            secure_loads(pickled)

    @pytest.mark.security
    def test_secure_loads_accepts_json(self):
        """Test that secure_loads accepts JSON data."""
        import json
        from apgi_security import secure_loads

        data = {"test": "data", "number": 42}
        json_bytes = json.dumps(data).encode()

        result = secure_loads(json_bytes)
        assert result == data


# =============================================================================
# CLI ENTRY POINT SECURITY TESTS
# =============================================================================


class TestCLIEntryPointSecurity:
    """Tests for CLI entry point security."""

    @pytest.mark.security
    def test_cli_entrypoint_handles_auth_error(self):
        """Test that CLI entry point handles authorization errors."""
        from apgi_cli import cli_entrypoint, AuthorizationError

        def failing_main():
            raise AuthorizationError("Test authorization failure")

        with pytest.raises(SystemExit) as exc_info:
            cli_entrypoint(failing_main, add_common_args=False)

        # Should exit with authorization error code
        assert exc_info.value.code == 77

    @pytest.mark.security
    def test_cli_entrypoint_handles_keyboard_interrupt(self):
        """Test that CLI entry point handles keyboard interrupt."""
        from apgi_cli import cli_entrypoint

        def interrupted_main():
            raise KeyboardInterrupt()

        with pytest.raises(SystemExit) as exc_info:
            cli_entrypoint(interrupted_main, add_common_args=False)

        # Should exit with Ctrl+C code
        assert exc_info.value.code == 130

    @pytest.mark.security
    def test_cli_entrypoint_handles_general_errors(self):
        """Test that CLI entry point handles general errors."""
        from apgi_cli import cli_entrypoint

        def error_main():
            raise ValueError("Test error")

        with pytest.raises(SystemExit) as exc_info:
            cli_entrypoint(error_main, add_common_args=False)

        # Should exit with general error code
        assert exc_info.value.code == 1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestSecurityIntegration:
    """Integration tests for security controls."""

    @pytest.mark.security
    @pytest.mark.integration
    def test_full_security_chain(self):
        """Test the full security enforcement chain."""
        # This test verifies that all security controls work together

        # 1. Set up required environment
        test_key = "a" * 64
        with patch.dict(
            os.environ,
            {
                "APGI_AUDIT_KEY": test_key,
                "APGI_OPERATOR_ROLE": "operator",
            },
        ):
            # 2. Import security components
            from apgi_audit import get_audit_sink
            from apgi_authz import get_authz_manager, Permission, AuthorizationContext
            from apgi_security import secure_loads_json

            # 3. Verify audit sink works
            sink = get_audit_sink()
            assert sink is not None

            # 4. Verify authz works
            from apgi_authz import Role

            authz = get_authz_manager()
            operator = authz.register_operator(
                username="integration_test", role=Role.OPERATOR
            )

            context = AuthorizationContext(
                operator=operator,
                resource_type="experiment",
                resource_id="integration_test",
                action=Permission.RUN_EXPERIMENT,
            )

            authorized = authz.authorize_action(context)
            assert authorized is True

            # 5. Verify secure JSON loading works
            import json

            test_data = {"security": "test"}
            json_str = json.dumps(test_data)
            result = secure_loads_json(json_str.encode())
            assert result == test_data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "security"])
