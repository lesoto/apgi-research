"""
Comprehensive tests for apgi_compliance.py - Compliance and data retention module.
"""

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from utils.apgi_compliance import (
    RETENTION_POLICIES,
    ComplianceManager,
    DataClassification,
    RetentionPolicy,
    pseudonymize_participant,
)


class TestDataClassification:
    """Tests for DataClassification enum."""

    def test_enum_values(self):
        """Test DataClassification enum values."""
        assert (
            DataClassification.PUBLIC.value == "public"
        )  # nosec: B101 - Test assertion
        assert (
            DataClassification.INTERNAL.value == "internal"
        )  # nosec: B101 - Test assertion
        assert (
            DataClassification.CONFIDENTIAL.value == "confidential"
        )  # nosec: B101 - Test assertion
        assert (
            DataClassification.RESTRICTED.value == "restricted"
        )  # nosec: B101 - Test assertion

    def test_all_classifications_present(self):
        """Test all expected classifications are present."""
        classifications = list(DataClassification)
        assert len(classifications) == 4  # nosec: B101 - Test assertion
        assert (
            DataClassification.PUBLIC in classifications
        )  # nosec: B101 - Test assertion
        assert (
            DataClassification.INTERNAL in classifications
        )  # nosec: B101 - Test assertion
        assert (
            DataClassification.CONFIDENTIAL in classifications
        )  # nosec: B101 - Test assertion
        assert (
            DataClassification.RESTRICTED in classifications
        )  # nosec: B101 - Test assertion


class TestRetentionPolicy:
    """Tests for RetentionPolicy dataclass."""

    def test_retention_policy_creation(self):
        """Test RetentionPolicy creation."""
        policy = RetentionPolicy(
            classification=DataClassification.CONFIDENTIAL,
            ttl_days=90,
            deletion_routine="secure_erase",
        )
        assert (
            policy.classification == DataClassification.CONFIDENTIAL
        )  # nosec: B101 - Test assertion
        assert policy.ttl_days == 90  # nosec: B101 - Test assertion
        assert policy.deletion_routine == "secure_erase"  # nosec: B101 - Test assertion

    def test_retention_policy_public(self):
        """Test public retention policy."""
        policy = RETENTION_POLICIES[DataClassification.PUBLIC]
        assert (
            policy.classification == DataClassification.PUBLIC
        )  # nosec: B101 - Test assertion
        assert policy.ttl_days == 3650  # nosec: B101 - Test assertion
        assert policy.deletion_routine == "archive"  # nosec: B101 - Test assertion

    def test_retention_policy_internal(self):
        """Test internal retention policy."""
        policy = RETENTION_POLICIES[DataClassification.INTERNAL]
        assert (
            policy.classification == DataClassification.INTERNAL
        )  # nosec: B101 - Test assertion
        assert policy.ttl_days == 365  # nosec: B101 - Test assertion
        assert policy.deletion_routine == "soft_delete"  # nosec: B101 - Test assertion

    def test_retention_policy_confidential(self):
        """Test confidential retention policy."""
        policy = RETENTION_POLICIES[DataClassification.CONFIDENTIAL]
        assert (
            policy.classification == DataClassification.CONFIDENTIAL
        )  # nosec: B101 - Test assertion
        assert policy.ttl_days == 90  # nosec: B101 - Test assertion
        assert policy.deletion_routine == "secure_erase"  # nosec: B101 - Test assertion

    def test_retention_policy_restricted(self):
        """Test restricted retention policy."""
        policy = RETENTION_POLICIES[DataClassification.RESTRICTED]
        assert (
            policy.classification == DataClassification.RESTRICTED
        )  # nosec: B101 - Test assertion
        assert policy.ttl_days == 30  # nosec: B101 - Test assertion
        assert policy.deletion_routine == "crypto_shred"  # nosec: B101 - Test assertion


class TestComplianceManager(unittest.TestCase):
    """Tests for ComplianceManager class."""

    def test_init(self):
        """Test ComplianceManager initialization."""
        manager = ComplianceManager()
        assert manager.audit_trail == []  # nosec: B101 - Test assertion

    def test_log_parameter_change(self):
        """Test logging parameter changes."""
        manager = ComplianceManager()
        manager.log_parameter_change("user1", "tau_s", 0.3, 0.4)

        assert len(manager.audit_trail) == 1  # nosec: B101 - Test assertion
        entry = manager.audit_trail[0]
        assert entry["action"] == "parameter_change"  # nosec: B101 - Test assertion
        assert entry["user"] == "user1"  # nosec: B101 - Test assertion
        assert entry["param_name"] == "tau_s"  # nosec: B101 - Test assertion
        assert entry["old_value"] == 0.3  # nosec: B101 - Test assertion
        assert entry["new_value"] == 0.4  # nosec: B101 - Test assertion
        assert "timestamp" in entry  # nosec: B101 - Test assertion

    def test_log_experiment_run(self):
        """Test logging experiment runs."""
        manager = ComplianceManager()
        manager.log_experiment_run("exp_123", DataClassification.INTERNAL)

        assert len(manager.audit_trail) == 1  # nosec: B101 - Test assertion
        entry = manager.audit_trail[0]
        assert entry["action"] == "experiment_run"  # nosec: B101 - Test assertion
        assert entry["experiment_id"] == "exp_123"  # nosec: B101 - Test assertion
        assert entry["classification"] == "internal"  # nosec: B101 - Test assertion
        assert "timestamp" in entry  # nosec: B101 - Test assertion

    def test_log_multiple_events(self):
        """Test logging multiple events."""
        manager = ComplianceManager()
        manager.log_parameter_change("user1", "param1", 1, 2)
        manager.log_experiment_run("exp1", DataClassification.PUBLIC)
        manager.log_parameter_change("user2", "param2", "a", "b")

        assert len(manager.audit_trail) == 3  # nosec: B101 - Test assertion

    @patch("utils.apgi_compliance.get_logger")
    def test_log_parameter_change_logs_to_logger(self, mock_get_logger):
        """Test that parameter changes are logged."""
        mock_logger = mock_get_logger.return_value
        manager = ComplianceManager()
        manager.log_parameter_change("user1", "tau_s", 0.3, 0.4)

        # Verify that get_logger was called with the correct name
        mock_get_logger.assert_called_once_with("apgi.compliance.manager")
        # Verify that the logger's info method was called once
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "Compliance Audit" in log_message  # nosec: B101 - Test assertion
        assert "parameter_change" in log_message  # nosec: B101 - Test assertion

    @patch("utils.apgi_compliance.get_logger")
    def test_log_experiment_run_logs_to_logger(self, mock_get_logger):
        """Test that experiment runs are logged."""
        mock_logger = mock_get_logger.return_value
        manager = ComplianceManager()
        manager.log_experiment_run("exp_123", DataClassification.INTERNAL)

        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "Compliance Audit" in log_message  # nosec: B101 - Test assertion
        assert "experiment_run" in log_message  # nosec: B101 - Test assertion


class TestComplianceManagerEnforceRetention:
    """Tests for ComplianceManager.enforce_retention method."""

    def test_enforce_retention_empty_list(self):
        """Test enforce_retention with empty list."""
        manager = ComplianceManager()
        result = manager.enforce_retention([])
        assert result == []  # nosec: B101 - Test assertion

    def test_enforce_retention_valid_records(self):
        """Test enforce_retention with valid records."""
        manager = ComplianceManager()
        current_time = datetime.now(timezone.utc)
        records = [
            {
                "id": "rec1",
                "classification": DataClassification.INTERNAL,
                "created_at": current_time.isoformat(),
            },
            {
                "id": "rec2",
                "classification": DataClassification.PUBLIC,
                "created_at": current_time.isoformat(),
            },
        ]
        result = manager.enforce_retention(records)
        assert len(result) == 2  # nosec: B101 - Test assertion
        assert result[0]["id"] == "rec1"  # nosec: B101 - Test assertion
        assert result[1]["id"] == "rec2"  # nosec: B101 - Test assertion

    def test_enforce_retention_expired_records_internal(self):
        """Test enforce_retention removes expired internal records."""
        manager = ComplianceManager()
        old_time = datetime.now(timezone.utc) - timedelta(days=400)
        records = [
            {
                "id": "rec1",
                "classification": DataClassification.INTERNAL,
                "created_at": old_time.isoformat(),
            },
        ]
        result = manager.enforce_retention(records)
        assert len(result) == 0  # nosec: B101 - Test assertion

    def test_enforce_retention_expired_records_confidential(self):
        """Test enforce_retention removes expired confidential records."""
        manager = ComplianceManager()
        old_time = datetime.now(timezone.utc) - timedelta(days=100)
        records = [
            {
                "id": "rec1",
                "classification": DataClassification.CONFIDENTIAL,
                "created_at": old_time.isoformat(),
            },
        ]
        result = manager.enforce_retention(records)
        assert len(result) == 0  # nosec: B101 - Test assertion

    def test_enforce_retention_expired_records_restricted(self):
        """Test enforce_retention removes expired restricted records."""
        manager = ComplianceManager()
        old_time = datetime.now() - timedelta(days=40)
        records = [
            {
                "id": "rec1",
                "classification": DataClassification.RESTRICTED,
                "created_at": old_time.isoformat(),
            },
        ]
        result = manager.enforce_retention(records)
        assert len(result) == 0  # nosec: B101 - Test assertion

    def test_enforce_retention_mixed_records(self):
        """Test enforce_retention with mix of valid and expired records."""
        manager = ComplianceManager()
        current_time = datetime.now()
        old_time = datetime.now() - timedelta(days=400)

        records = [
            {
                "id": "valid",
                "classification": DataClassification.INTERNAL,
                "created_at": current_time.isoformat(),
            },
            {
                "id": "expired",
                "classification": DataClassification.INTERNAL,
                "created_at": old_time.isoformat(),
            },
        ]
        result = manager.enforce_retention(records)
        assert len(result) == 1  # nosec: B101 - Test assertion
        assert result[0]["id"] == "valid"  # nosec: B101 - Test assertion

    def test_enforce_retention_string_classification(self):
        """Test enforce_retention with string classification."""
        manager = ComplianceManager()
        current_time = datetime.now()
        records = [
            {
                "id": "rec1",
                "classification": "internal",
                "created_at": current_time.isoformat(),
            },
        ]
        result = manager.enforce_retention(records)
        assert len(result) == 1  # nosec: B101 - Test assertion

    def test_enforce_retention_invalid_classification(self):
        """Test enforce_retention with invalid classification."""
        manager = ComplianceManager()
        current_time = datetime.now()
        records = [
            {
                "id": "rec1",
                "classification": "invalid_classification",
                "created_at": current_time.isoformat(),
            },
        ]
        result = manager.enforce_retention(records)
        # Should default to INTERNAL and keep record
        assert len(result) == 1  # nosec: B101 - Test assertion

    def test_enforce_retention_no_classification(self):
        """Test enforce_retention with no classification (defaults to INTERNAL)."""
        manager = ComplianceManager()
        current_time = datetime.now()
        records = [
            {
                "id": "rec1",
                "created_at": current_time.isoformat(),
            },
        ]
        result = manager.enforce_retention(records)
        assert len(result) == 1  # nosec: B101 - Test assertion

    def test_enforce_retention_no_created_at(self):
        """Test enforce_retention with no created_at (uses current time)."""
        manager = ComplianceManager()
        records = [
            {
                "id": "rec1",
                "classification": DataClassification.INTERNAL,
            },
        ]
        result = manager.enforce_retention(records)
        assert len(result) == 1  # nosec: B101 - Test assertion

    @patch("utils.apgi_logging.get_logger")
    def test_enforce_retention_deletion_logged(self, mock_get_logger):
        """Test that deletion routine is logged."""
        mock_logger = mock_get_logger.return_value
        manager = ComplianceManager()
        # Replace the manager's logger with our mock
        manager.logger = mock_logger

        old_time = datetime.now(timezone.utc) - timedelta(days=100)
        records = [
            {
                "id": "rec1",
                "classification": DataClassification.CONFIDENTIAL,
                "created_at": old_time.isoformat(),
            },
        ]
        manager.enforce_retention(records)

        mock_logger.info.assert_called()
        log_message = mock_logger.info.call_args[0][0]
        assert "secure_erase" in log_message  # nosec: B101 - Test assertion


class TestPseudonymizeParticipant:
    """Tests for pseudonymize_participant function."""

    def setup_method(self):
        """Set up test environment with required environment variable."""
        # Set the required environment variable for pseudonymization
        import os

        os.environ["APGI_PSEUDONYM_SALT"] = "test_salt_for_pseudonymization"

    def test_pseudonymize_basic(self):
        """Test basic pseudonymization."""
        participant_id = "user123"
        result = pseudonymize_participant(participant_id)
        assert result != participant_id  # nosec: B101 - Test assertion
        assert len(result) == 64  # SHA-256 hex length  # nosec: B101 - Test assertion

    def test_pseudonymize_consistency(self):
        """Test that same input produces same output."""
        participant_id = "user123"
        result1 = pseudonymize_participant(participant_id)
        result2 = pseudonymize_participant(participant_id)
        assert result1 == result2  # nosec: B101 - Test assertion

    def test_pseudonymize_different_inputs(self):
        """Test that different inputs produce different outputs."""
        result1 = pseudonymize_participant("user1")
        result2 = pseudonymize_participant("user2")
        assert result1 != result2  # nosec: B101 - Test assertion

    def test_pseudonymize_empty_string(self):
        """Test pseudonymization of empty string."""
        result = pseudonymize_participant("")
        assert result == ""  # nosec: B101 - Test assertion

    def test_pseudonymize_custom_salt(self):
        """Test pseudonymization with custom salt."""
        participant_id = "user123"
        result1 = pseudonymize_participant(participant_id, "salt1")
        result2 = pseudonymize_participant(participant_id, "salt2")
        assert result1 != result2  # nosec: B101 - Test assertion

    def test_pseudonymize_with_default_salt(self):
        """Test pseudonymization uses default salt."""
        participant_id = "user123"
        result = pseudonymize_participant(participant_id)
        expected = pseudonymize_participant(
            participant_id, "test_salt_for_pseudonymization"
        )
        assert result == expected  # nosec: B101 - Test assertion


class TestComplianceManagerIntegration:
    """Integration tests for ComplianceManager."""

    def test_full_audit_workflow(self):
        """Test complete audit workflow."""
        manager = ComplianceManager()

        # Log parameter changes
        manager.log_parameter_change("admin", "tau_s", 0.3, 0.35)
        manager.log_parameter_change("admin", "beta", 1.0, 1.5)

        # Log experiment runs
        manager.log_experiment_run("exp_1", DataClassification.INTERNAL)
        manager.log_experiment_run("exp_2", DataClassification.CONFIDENTIAL)

        assert len(manager.audit_trail) == 4  # nosec: B101 - Test assertion

        # Check parameter changes
        param_changes = [
            e for e in manager.audit_trail if e["action"] == "parameter_change"
        ]
        assert len(param_changes) == 2  # nosec: B101 - Test assertion

        # Check experiment runs
        exp_runs = [e for e in manager.audit_trail if e["action"] == "experiment_run"]
        assert len(exp_runs) == 2  # nosec: B101 - Test assertion

    def test_retention_with_audit(self):
        """Test retention enforcement with audit trail."""
        manager = ComplianceManager()

        current_time = datetime.now()
        old_time = datetime.now() - timedelta(days=100)

        # Create records with different ages
        records = [
            {
                "id": "current_conf",
                "classification": DataClassification.CONFIDENTIAL,
                "created_at": current_time.isoformat(),
            },
            {
                "id": "old_conf",
                "classification": DataClassification.CONFIDENTIAL,
                "created_at": old_time.isoformat(),
            },
            {
                "id": "old_public",
                "classification": DataClassification.PUBLIC,
                "created_at": old_time.isoformat(),
            },
        ]

        result = manager.enforce_retention(records)

        # Should keep current confidential and old public (3650 day TTL)
        # Should remove old confidential (90 day TTL)
        assert len(result) == 2  # nosec: B101 - Test assertion
        ids = [r["id"] for r in result]
        assert "current_conf" in ids  # nosec: B101 - Test assertion
        assert "old_public" in ids  # nosec: B101 - Test assertion
        assert "old_conf" not in ids  # nosec: B101 - Test assertion


class TestComplianceEdgeCases:
    """Edge case tests for compliance module."""

    def test_retention_policy_missing(self):
        """Test enforce_retention with unknown classification."""
        manager = ComplianceManager()
        current_time = datetime.now()

        records = [
            {
                "id": "rec1",
                "classification": "UNKNOWN",
                "created_at": current_time.isoformat(),
            },
        ]
        result = manager.enforce_retention(records)
        # Should default to INTERNAL and keep record
        assert len(result) == 1  # nosec: B101 - Test assertion

    def test_pseudonymize_unicode(self):
        """Test pseudonymization with unicode characters."""
        participant_id = "用户123"
        result = pseudonymize_participant(participant_id)
        assert len(result) == 64  # nosec: B101 - Test assertion
        assert result != participant_id  # nosec: B101 - Test assertion

    def test_pseudonymize_long_id(self):
        """Test pseudonymization with very long ID."""
        participant_id = "a" * 10000
        result = pseudonymize_participant(participant_id)
        assert len(result) == 64  # nosec: B101 - Test assertion

    def test_pseudonymize_special_characters(self):
        """Test pseudonymization with special characters."""
        participant_id = "user@example.com|test+123"
        result = pseudonymize_participant(participant_id)
        assert len(result) == 64  # nosec: B101 - Test assertion
        assert result != participant_id  # nosec: B101 - Test assertion
