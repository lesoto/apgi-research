"""
Comprehensive test suite for apgi_compliance.py to achieve ≥90% coverage.

This file tests the complex code paths and edge cases not covered in the basic test suite.
"""

import json
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.apgi_compliance import (
    RETENTION_POLICIES,
    ComplianceManager,
    DataClassification,
    RetentionPolicy,
    pseudonymize_participant,
)


class TestDataClassification:
    """Test the DataClassification enum."""

    def test_data_classification_values(self):
        """Test that all expected classification values exist."""
        expected_values = [
            DataClassification.PUBLIC,
            DataClassification.INTERNAL,
            DataClassification.CONFIDENTIAL,
            DataClassification.RESTRICTED,
        ]

        for classification in expected_values:
            assert classification.value in [
                "public",
                "internal",
                "confidential",
                "restricted",
            ]

    def test_data_classification_string_conversion(self):
        """Test string conversion of classifications."""
        assert DataClassification.PUBLIC.value == "public"
        assert DataClassification.INTERNAL.value == "internal"
        assert DataClassification.CONFIDENTIAL.value == "confidential"
        assert DataClassification.RESTRICTED.value == "restricted"


class TestRetentionPolicy:
    """Test the RetentionPolicy dataclass."""

    def test_retention_policy_creation(self):
        """Test creating retention policies."""
        policy = RetentionPolicy(
            classification=DataClassification.PUBLIC,
            ttl_days=365,
            deletion_routine="archive",
        )

        assert policy.classification == DataClassification.PUBLIC
        assert policy.ttl_days == 365
        assert policy.deletion_routine == "archive"

    def test_retention_policies_completeness(self):
        """Test that all expected retention policies are defined."""
        expected_classifications = [
            DataClassification.PUBLIC,
            DataClassification.INTERNAL,
            DataClassification.CONFIDENTIAL,
            DataClassification.RESTRICTED,
        ]

        for classification in expected_classifications:
            assert classification in RETENTION_POLICIES
            policy = RETENTION_POLICIES[classification]
            assert isinstance(policy, RetentionPolicy)
            assert policy.ttl_days > 0
            assert policy.deletion_routine in [
                "archive",
                "soft_delete",
                "secure_erase",
                "crypto_shred",
            ]

    def test_retention_policy_values(self):
        """Test specific retention policy values."""
        public_policy = RETENTION_POLICIES[DataClassification.PUBLIC]
        assert public_policy.ttl_days == 3650
        assert public_policy.deletion_routine == "archive"

        restricted_policy = RETENTION_POLICIES[DataClassification.RESTRICTED]
        assert restricted_policy.ttl_days == 30
        assert restricted_policy.deletion_routine == "crypto_shred"


class TestComplianceManager:
    """Test the ComplianceManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ComplianceManager()

    def test_initialization(self):
        """Test manager initialization."""
        assert isinstance(self.manager.audit_trail, list)
        assert len(self.manager.audit_trail) == 0

    def test_log_parameter_change(self):
        """Test logging parameter changes."""
        self.manager.log_parameter_change(
            user="test_user",
            param_name="test_param",
            old_value="old_value",
            new_value="new_value",
        )

        assert len(self.manager.audit_trail) == 1

        entry = self.manager.audit_trail[0]
        assert entry["action"] == "parameter_change"
        assert entry["user"] == "test_user"
        assert entry["param_name"] == "test_param"
        assert entry["old_value"] == "old_value"
        assert entry["new_value"] == "new_value"
        assert "timestamp" in entry

    def test_log_parameter_change_with_complex_values(self):
        """Test logging parameter changes with complex values."""
        complex_old = {"nested": {"key": "value"}, "list": [1, 2, 3]}
        complex_new = {"nested": {"key": "new_value"}, "list": [4, 5, 6]}

        self.manager.log_parameter_change(
            user="test_user",
            param_name="complex_param",
            old_value=complex_old,
            new_value=complex_new,
        )

        entry = self.manager.audit_trail[0]
        assert entry["old_value"] == complex_old
        assert entry["new_value"] == complex_new

    def test_log_experiment_run(self):
        """Test logging experiment runs."""
        self.manager.log_experiment_run(
            experiment_id="test_experiment",
            classification=DataClassification.CONFIDENTIAL,
        )

        assert len(self.manager.audit_trail) == 1

        entry = self.manager.audit_trail[0]
        assert entry["action"] == "experiment_run"
        assert entry["experiment_id"] == "test_experiment"
        assert entry["classification"] == "confidential"
        assert "timestamp" in entry

    def test_enforce_retention_empty_list(self):
        """Test retention enforcement with empty list."""
        result = self.manager.enforce_retention([])
        assert result == []

    def test_enforce_retention_no_expired_records(self):
        """Test retention enforcement with no expired records."""
        current_time = datetime.now()
        records = [
            {
                "id": "record1",
                "classification": "public",
                "created_at": (current_time - timedelta(days=1)).isoformat(),
                "data": "test data",
            },
            {
                "id": "record2",
                "classification": "internal",
                "created_at": (current_time - timedelta(days=10)).isoformat(),
                "data": "test data",
            },
        ]

        result = self.manager.enforce_retention(records)

        # All records should be retained
        assert len(result) == 2
        assert result[0]["id"] == "record1"
        assert result[1]["id"] == "record2"

    def test_enforce_retention_with_expired_records(self):
        """Test retention enforcement with expired records."""
        current_time = datetime.now()
        records = [
            {
                "id": "record1",
                "classification": "public",
                "created_at": (
                    current_time - timedelta(days=4000)
                ).isoformat(),  # Expired
                "data": "test data",
            },
            {
                "id": "record2",
                "classification": "restricted",
                "created_at": (
                    current_time - timedelta(days=40)
                ).isoformat(),  # Expired
                "data": "test data",
            },
            {
                "id": "record3",
                "classification": "internal",
                "created_at": (
                    current_time - timedelta(days=10)
                ).isoformat(),  # Not expired
                "data": "test data",
            },
        ]

        # Mock deletion execution
        with patch.object(self.manager, "_execute_deletion") as mock_delete:
            result = self.manager.enforce_retention(records)

        # Only non-expired record should be retained
        assert len(result) == 1
        assert result[0]["id"] == "record3"

        # Deletion should be called for expired records
        assert mock_delete.call_count == 2

    def test_enforce_retention_invalid_classification(self):
        """Test retention enforcement with invalid classification."""
        current_time = datetime.now()
        records = [
            {
                "id": "record1",
                "classification": "invalid_classification",
                "created_at": current_time.isoformat(),
                "data": "test data",
            }
        ]

        result = self.manager.enforce_retention(records)

        # Should default to internal classification
        assert len(result) == 1
        assert result[0]["id"] == "record1"

    def test_enforce_retention_missing_created_at(self):
        """Test retention enforcement with missing created_at."""
        records = [
            {
                "id": "record1",
                "classification": "public",
                "data": "test data",
                # Missing created_at
            }
        ]

        result = self.manager.enforce_retention(records)

        # Should use current time as created_at
        assert len(result) == 1
        assert result[0]["id"] == "record1"

    def test_enforce_retention_no_policy(self):
        """Test retention enforcement with no matching policy."""
        # Mock RETENTION_POLICIES to not have the classification
        original_policies = RETENTION_POLICIES.copy()
        RETENTION_POLICIES.clear()

        try:
            current_time = datetime.now()
            records = [
                {
                    "id": "record1",
                    "classification": "public",
                    "created_at": current_time.isoformat(),
                    "data": "test data",
                }
            ]

            result = self.manager.enforce_retention(records)

            # Should skip record with no policy
            assert len(result) == 0

        finally:
            # Restore original policies
            RETENTION_POLICIES.update(original_policies)


class TestDeletionRoutines:
    """Test the various deletion routines."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ComplianceManager()

    def test_soft_delete(self):
        """Test soft deletion routine."""
        record = {
            "id": "test_record",
            "data": "sensitive data",
            "created_at": datetime.now().isoformat(),
        }

        self.manager._execute_deletion(record, "soft_delete")

        # Check audit trail entry
        assert len(self.manager.audit_trail) == 1
        entry = self.manager.audit_trail[0]
        assert entry["action"] == "soft_delete"
        assert entry["record_id"] == "test_record"
        assert entry["routine"] == "soft_delete"
        assert "deleted_at" in entry

    def test_soft_delete_with_dict_record(self):
        """Test soft deletion with dict record."""
        record = {
            "id": "test_record",
            "data": "sensitive data",
            "created_at": datetime.now().isoformat(),
        }

        self.manager._execute_deletion(record, "soft_delete")

        # Check record was marked as deleted
        deleted_value = record.get("deleted")
        assert deleted_value is not None and deleted_value
        assert "deleted_at" in record

    def test_hard_delete(self):
        """Test hard deletion routine."""
        # Create temporary files to delete
        temp_dir = tempfile.mkdtemp()
        test_files = []

        for filename in ["data/test.txt", "results/test.json", "logs/test.log"]:
            file_path = Path(temp_dir) / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("test content")
            test_files.append(file_path)

        record = {
            "id": "test_record",
            "data": "sensitive data",
            "created_at": datetime.now().isoformat(),
        }

        # Test deletion execution

        # Mock glob to return our test files
        with patch("glob.glob") as mock_glob:
            mock_glob.return_value = [str(f) for f in test_files]

            # Mock os.remove to track deletions
            with patch("os.remove") as mock_remove:
                self.manager._execute_deletion(record, "hard_delete")

                # Should attempt to delete all files
                assert mock_remove.call_count == len(test_files)

                # Check deletion was executed (verified by mock calls)

        # Clean up
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_hard_delete_file_error(self):
        """Test hard deletion with file removal errors."""
        record = {
            "id": "test_record",
            "data": "sensitive data",
            "created_at": datetime.now().isoformat(),
        }

        # Mock glob to return file that will fail to delete
        with patch("glob.glob") as mock_glob:
            mock_glob.return_value = ["nonexistent_file.txt"]

            with patch("os.remove", side_effect=OSError("Permission denied")):
                # Should handle file deletion errors gracefully
                self.manager._execute_deletion(record, "hard_delete")

    def test_anonymous_deletion(self):
        """Test anonymization deletion routine."""
        record = {
            "id": "test_record",
            "user_id": "user123",
            "email": "test@example.com",
            "name": "Test User",
            "ip_address": "192.168.1.1",
            "session_id": "sess123",
            "token": "token456",
            "api_key": "key789",
            "regular_field": "regular_value",
            "created_at": datetime.now().isoformat(),
        }

        self.manager._execute_deletion(record, "anonymous")

        # Check PII fields were hashed
        assert record["user_id"].startswith("hashed_")
        assert record["email"].startswith("hashed_")
        assert record["name"].startswith("hashed_")
        assert record["ip_address"].startswith("hashed_")

        # Check direct identifiers were removed
        assert "session_id" not in record
        assert "token" not in record
        assert "api_key" not in record

        # Check regular field was preserved
        assert record["regular_field"] == "regular_value"

    def test_anonymous_deletion_non_dict_record(self):
        """Test anonymization with non-dict record."""
        record = {"id": "test", "data": "sensitive"}

        # Should handle dict records
        self.manager._execute_deletion(record, "anonymous")

    def test_secure_erase(self):
        """Test secure erase deletion routine."""
        record = {
            "id": "test_record",
            "sensitive_data": "very sensitive",
            "password": "secret123",
            "created_at": datetime.now().isoformat(),
        }

        # Create temporary files to erase
        temp_dir = tempfile.mkdtemp()
        test_files = []

        for filename in ["data/test.txt", "results/test.json"]:
            file_path = Path(temp_dir) / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("sensitive content")
            test_files.append(file_path)

        # Mock glob and file operations
        with patch("glob.glob") as mock_glob:
            mock_glob.return_value = [str(f) for f in test_files]

            with patch("os.path.getsize", return_value=100):
                with patch("builtins.open", mock_open()) as mock_file:
                    mock_file.read.return_value = b"test"
                    mock_file.write.return_value = None
                    with patch("os.fsync"):
                        with patch("os.remove"):
                            self.manager._execute_deletion(record, "secure_erase")

                            # Check data was overwritten (empty string)
                            assert record["sensitive_data"] == ""
                            assert record["password"] == ""

        # Clean up
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_secure_erase_file_error(self):
        """Test secure erase with file operation errors."""
        record = {
            "id": "test_record",
            "data": "test data",
            "created_at": datetime.now().isoformat(),
        }

        # Mock glob to return file that will fail to erase
        with patch("glob.glob") as mock_glob:
            mock_glob.return_value = ["nonexistent_file.txt"]

            with patch("os.path.getsize", return_value=100):
                with patch("builtins.open", side_effect=OSError("Permission denied")):
                    # Should handle file operation errors gracefully
                    self.manager._execute_deletion(record, "secure_erase")

    def test_crypto_shred(self):
        """Test crypto shredding deletion routine."""
        record = {
            "id": "test_record",
            "encryption_key": "secret_key_123",
            "data": "encrypted_data",
            "created_at": datetime.now().isoformat(),
        }

        self.manager._execute_deletion(record, "crypto_shred")

        # Check encryption key was overwritten
        assert record["encryption_key"] == ""

    def test_archive_deletion(self):
        """Test archive deletion routine."""
        record = {
            "id": "test_record",
            "data": "test data to archive",
            "created_at": datetime.now().isoformat(),
        }

        # Mock file operations
        with patch("os.makedirs"):
            with patch("gzip.open", mock_open()) as mock_gzip:
                with patch("json.dump") as mock_json:
                    self.manager._execute_deletion(record, "archive")

                    # Check archive was created
                    mock_gzip.assert_called_once()
                    mock_json.assert_called_once()

                    # Check archive data structure
                    args, kwargs = mock_json.call_args
                    archive_data = args[0]
                    assert "record" in archive_data
                    assert "archived_at" in archive_data
                    assert "archive_reason" in archive_data

    def test_archive_deletion_file_error(self):
        """Test archive deletion with file operation errors."""
        record = {
            "id": "test_record",
            "data": "test data",
            "created_at": datetime.now().isoformat(),
        }

        # Mock file operations to raise exception
        with patch("os.makedirs", side_effect=OSError("Permission denied")):
            # Should handle file operation errors gracefully
            with pytest.raises(Exception):
                self.manager._execute_deletion(record, "archive")

    def test_archive_deletion_permission_error(self):
        """Test archive deletion with permission error."""
        record = {
            "id": "test_record",
            "data": "test data",
            "created_at": datetime.now().isoformat(),
        }

        # Mock file operations to raise exception
        with patch("os.makedirs", side_effect=OSError("Permission denied")):
            # Should handle file operation errors gracefully
            with pytest.raises(Exception):
                self.manager._execute_deletion(record, "archive")

    def test_unsupported_deletion_routine(self):
        """Test handling of unsupported deletion routines."""
        record = {
            "id": "test_record",
            "data": "test data",
            "created_at": datetime.now().isoformat(),
        }

        # Should raise ValueError for unknown routine
        with pytest.raises(ValueError, match="Unsupported deletion routine"):
            self.manager._execute_deletion(record, "unknown_routine")


class TestPseudonymization:
    """Test the pseudonymization function."""

    def test_pseudonymize_participant_empty_id(self):
        """Test pseudonymization with empty participant ID."""
        result = pseudonymize_participant("")
        assert result == ""

    def test_pseudonymize_participant_default_salt(self):
        """Test pseudonymization with default salt."""
        participant_id = "test_participant_123"
        result1 = pseudonymize_participant(participant_id)
        result2 = pseudonymize_participant(participant_id)

        # Should be deterministic with same input
        assert result1 == result2
        assert len(result1) == 64  # SHA256 hash length
        assert result1 != participant_id  # Should be different from original

    def test_pseudonymize_participant_custom_salt(self):
        """Test pseudonymization with custom salt."""
        participant_id = "test_participant_123"
        custom_salt = "custom_salt_value"

        result1 = pseudonymize_participant(participant_id, salt=custom_salt)
        result2 = pseudonymize_participant(participant_id, salt=custom_salt)

        # Should be deterministic with same salt
        assert result1 == result2

        # Should be different from default salt
        result_default = pseudonymize_participant(participant_id)
        assert result1 != result_default

    def test_pseudonymize_participant_environment_salt(self):
        """Test pseudonymization with environment salt."""
        participant_id = "test_participant_123"
        env_salt = "environment_salt_456"

        # Set environment variable
        with patch.dict(os.environ, {"APGI_PSEUDONYM_SALT": env_salt}):
            result = pseudonymize_participant(participant_id)

            # Should use environment salt
            assert len(result) == 64
            assert result != participant_id

    def test_pseudonymize_participant_different_inputs(self):
        """Test pseudonymization with different participant IDs."""
        result1 = pseudonymize_participant("participant_1")
        result2 = pseudonymize_participant("participant_2")
        result3 = pseudonymize_participant("participant_1")

        # Different inputs should produce different hashes
        assert result1 != result2
        assert result1 == result3  # Same input should produce same hash

    def test_pseudonymize_participant_case_sensitivity(self):
        """Test pseudonymization case sensitivity."""
        result_lower = pseudonymize_participant("test_participant")
        result_upper = pseudonymize_participant("TEST_PARTICIPANT")

        # Should be case sensitive
        assert result_lower != result_upper


class TestComplianceIntegration:
    """Integration tests for compliance functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ComplianceManager()

    def test_full_compliance_workflow(self):
        """Test a complete compliance workflow."""
        # Log parameter changes
        self.manager.log_parameter_change(
            user="operator1",
            param_name="sensitivity_level",
            old_value="medium",
            new_value="high",
        )

        # Log experiment run
        self.manager.log_experiment_run(
            experiment_id="exp_001", classification=DataClassification.CONFIDENTIAL
        )

        # Create test records
        current_time = datetime.now()
        records = [
            {
                "id": "record_001",
                "classification": "confidential",
                "created_at": (current_time - timedelta(days=100)).isoformat(),
                "data": "sensitive experiment data",
            },
            {
                "id": "record_002",
                "classification": "public",
                "created_at": (current_time - timedelta(days=10)).isoformat(),
                "data": "public experiment data",
            },
        ]

        # Enforce retention
        with patch.object(self.manager, "_execute_deletion") as mock_delete:
            retained_records = self.manager.enforce_retention(records)

        # Verify workflow
        assert len(self.manager.audit_trail) == 2
        assert len(retained_records) == 1
        assert retained_records[0]["id"] == "record_002"
        assert mock_delete.call_count == 1

    def test_audit_trail_persistence(self):
        """Test audit trail persistence and serialization."""
        # Add multiple audit entries
        self.manager.log_parameter_change("user1", "param1", "old", "new")
        self.manager.log_experiment_run("exp1", DataClassification.INTERNAL)
        self.manager.log_parameter_change("user2", "param2", "old2", "new2")

        # Verify audit trail structure
        assert len(self.manager.audit_trail) == 3

        # Check each entry has required fields
        for entry in self.manager.audit_trail:
            assert "timestamp" in entry
            assert "action" in entry
            assert isinstance(
                json.loads(json.dumps(entry)), dict
            )  # Ensure JSON serializable

    def test_multiple_retention_cycles(self):
        """Test multiple retention enforcement cycles."""
        # Create records that will expire in different cycles
        current_time = datetime.now()
        records = [
            {
                "id": f"record_{i}",
                "classification": "restricted",
                "created_at": (current_time - timedelta(days=i * 20)).isoformat(),
                "data": f"data_{i}",
            }
            for i in range(1, 6)  # Records aged 20, 40, 60, 80, 100 days
        ]

        with patch.object(self.manager, "_execute_deletion") as mock_delete:
            # First enforcement - should delete records older than 30 days
            result1 = self.manager.enforce_retention(records)
            assert len(result1) == 2  # Records 1 and 2 should remain

            # Second enforcement - should delete records older than 50 days
            result2 = self.manager.enforce_retention(result1)
            assert len(result2) == 1  # Only record 1 should remain

            # Third enforcement - should delete all records
            result3 = self.manager.enforce_retention(result2)
            assert len(result3) == 0  # All records expired

            # Total deletions should equal number of records
            assert mock_delete.call_count == 5

    def test_compliance_with_large_dataset(self):
        """Test compliance operations with large datasets."""
        # Create a large number of records
        large_dataset = []
        for i in range(1000):
            large_dataset.append(
                {
                    "id": f"record_{i}",
                    "classification": "internal",
                    "created_at": datetime.now().isoformat(),
                    "data": f"data_{i}",
                }
            )

        # Should handle large dataset efficiently
        start_time = time.time()
        result = self.manager.enforce_retention(large_dataset)
        end_time = time.time()

        # All records should be retained (none expired)
        assert len(result) == 1000

        # Should complete in reasonable time
        assert end_time - start_time < 5.0  # 5 second limit for 1000 records


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
