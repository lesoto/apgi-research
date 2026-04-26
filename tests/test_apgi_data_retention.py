"""
Comprehensive tests for apgi_data_retention.py module.
Aiming for 100% code coverage.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from apgi_data_retention import (
    DataSubjectRecord,
    DeletionExecutor,
    RetentionConfig,
    RetentionJobScheduler,
    RetentionPolicy,
    _retention_scheduler,
    get_retention_scheduler,
    set_retention_scheduler,
)


class TestRetentionPolicy:
    """Test RetentionPolicy enum."""

    def test_retention_policy_values(self):
        """Test all retention policy values."""
        assert RetentionPolicy.PERMANENT.value == "permanent"
        assert RetentionPolicy.GDPR_DEFAULT.value == "gdpr_default"
        assert RetentionPolicy.CCPA_DEFAULT.value == "ccpa_default"
        assert RetentionPolicy.HIPAA_DEFAULT.value == "hipaa_default"
        assert RetentionPolicy.CUSTOM.value == "custom"


class TestRetentionConfig:
    """Test RetentionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = RetentionConfig()
        assert config.policy == RetentionPolicy.GDPR_DEFAULT
        assert config.retention_days == 1095
        assert config.auto_delete_enabled is True
        assert config.deletion_verification_required is True
        assert config.audit_trail_retention_days == 2555

    def test_custom_config(self):
        """Test custom configuration."""
        config = RetentionConfig(
            policy=RetentionPolicy.CUSTOM,
            retention_days=365,
            auto_delete_enabled=False,
        )
        assert config.policy == RetentionPolicy.CUSTOM
        assert config.retention_days == 365
        assert config.auto_delete_enabled is False

    def test_get_retention_period(self):
        """Test retention period calculation."""
        config = RetentionConfig(retention_days=30)
        period = config.get_retention_period()
        assert period == timedelta(days=30)

    def test_get_audit_retention_period(self):
        """Test audit retention period calculation."""
        config = RetentionConfig(audit_trail_retention_days=365)
        period = config.get_audit_retention_period()
        assert period == timedelta(days=365)


class TestDataSubjectRecord:
    """Test DataSubjectRecord dataclass."""

    def test_default_creation(self):
        """Test default record creation."""
        record = DataSubjectRecord(
            subject_id="test_001",
            subject_name="Test User",
        )
        assert record.subject_id == "test_001"
        assert record.subject_name == "Test User"
        assert record.data_categories == []
        assert record.retention_policy == RetentionPolicy.GDPR_DEFAULT
        assert record.deletion_requested is False
        assert record.deletion_completed is False
        assert isinstance(record.created_at, datetime)
        assert isinstance(record.last_accessed, datetime)

    def test_is_retention_expired_permanent(self):
        """Test that permanent policy never expires."""
        record = DataSubjectRecord(
            subject_id="test_001",
            subject_name="Test",
            retention_policy=RetentionPolicy.PERMANENT,
            created_at=datetime.utcnow() - timedelta(days=10000),
        )
        config = RetentionConfig()
        assert record.is_retention_expired(config) is False

    def test_is_retention_expired_expired(self):
        """Test expired retention detection."""
        record = DataSubjectRecord(
            subject_id="test_001",
            subject_name="Test",
            retention_policy=RetentionPolicy.GDPR_DEFAULT,
            created_at=datetime.utcnow() - timedelta(days=2000),
        )
        config = RetentionConfig(retention_days=1095)
        assert record.is_retention_expired(config) is True

    def test_is_retention_expired_not_expired(self):
        """Test non-expired retention."""
        record = DataSubjectRecord(
            subject_id="test_001",
            subject_name="Test",
            retention_policy=RetentionPolicy.GDPR_DEFAULT,
            created_at=datetime.utcnow() - timedelta(days=100),
        )
        config = RetentionConfig(retention_days=1095)
        assert record.is_retention_expired(config) is False

    def test_mark_for_deletion(self):
        """Test marking record for deletion."""
        record = DataSubjectRecord(
            subject_id="test_001",
            subject_name="Test",
        )
        record.mark_for_deletion()
        assert record.deletion_requested is True
        assert isinstance(record.deletion_requested_at, datetime)

    def test_mark_deletion_complete(self):
        """Test marking deletion as complete."""
        record = DataSubjectRecord(
            subject_id="test_001",
            subject_name="Test",
        )
        record.mark_deletion_complete()
        assert record.deletion_completed is True
        assert isinstance(record.deletion_completed_at, datetime)


class TestDeletionExecutor:
    """Test DeletionExecutor class."""

    @pytest.fixture
    def mock_audit_sink(self):
        """Create mock audit sink."""
        return Mock()

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return Mock()

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RetentionConfig()

    @pytest.fixture
    def executor(self, mock_audit_sink, mock_logger, config):
        """Create deletion executor with mocks."""
        with patch("apgi_data_retention.get_logger", return_value=mock_logger):
            with patch(
                "apgi_data_retention.get_audit_sink", return_value=mock_audit_sink
            ):
                return DeletionExecutor(config)

    def test_delete_experiment_data_success(self, executor, mock_audit_sink, tmp_path):
        """Test successful experiment data deletion."""
        # Create a test file
        test_file = tmp_path / "test_experiment.npy"
        test_file.write_text("test data")

        result = executor.delete_experiment_data(
            subject_id="sub_001",
            experiment_id="exp_001",
            data_path=str(test_file),
        )

        assert result is True
        assert not test_file.exists()
        mock_audit_sink.record_event.assert_called_once()
        call_args = mock_audit_sink.record_event.call_args[1]
        assert call_args["event_type"].value == "data_deleted"
        assert call_args["status"] == "success"

    def test_delete_experiment_data_no_path(self, executor, mock_audit_sink):
        """Test deletion without file path."""
        result = executor.delete_experiment_data(
            subject_id="sub_001",
            experiment_id="exp_001",
            data_path=None,
        )

        assert result is True
        mock_audit_sink.record_event.assert_called_once()

    def test_delete_experiment_data_failure(self, executor, mock_audit_sink, tmp_path):
        """Test failed experiment data deletion."""
        # Try to delete non-existent file in non-existent directory
        result = executor.delete_experiment_data(
            subject_id="sub_001",
            experiment_id="exp_001",
            data_path="/nonexistent/path/file.npy",
        )

        assert result is False
        mock_audit_sink.record_event.assert_called_once()
        call_args = mock_audit_sink.record_event.call_args[1]
        assert call_args["status"] == "failure"

    def test_delete_config_data_success(self, executor, mock_audit_sink):
        """Test successful config data deletion."""
        result = executor.delete_config_data(
            subject_id="sub_001",
            config_id="config_001",
        )

        assert result is True
        mock_audit_sink.record_event.assert_called_once()
        call_args = mock_audit_sink.record_event.call_args[1]
        assert call_args["event_type"].value == "data_deleted"
        assert call_args["resource_type"] == "config"

    def test_delete_config_data_failure(self, executor, mock_audit_sink):
        """Test failed config data deletion."""
        mock_audit_sink.record_event.side_effect = Exception("Audit failure")

        result = executor.delete_config_data(
            subject_id="sub_001",
            config_id="config_001",
        )

        assert result is False

    def test_destroy_kms_key_with_callback_success(self, executor, mock_audit_sink):
        """Test KMS key destruction with callback."""
        mock_callback = Mock(return_value=True)

        result = executor.destroy_kms_key(
            subject_id="sub_001",
            key_id="key_001",
            kms_callback=mock_callback,
        )

        assert result is True
        mock_callback.assert_called_once_with("key_001")
        mock_audit_sink.record_event.assert_called_once()

    def test_destroy_kms_key_callback_failure(self, executor, mock_audit_sink):
        """Test KMS key destruction with failing callback."""
        mock_callback = Mock(return_value=False)

        result = executor.destroy_kms_key(
            subject_id="sub_001",
            key_id="key_001",
            kms_callback=mock_callback,
        )

        assert result is False

    def test_destroy_kms_key_no_callback(self, executor, mock_audit_sink):
        """Test KMS key destruction without callback."""
        result = executor.destroy_kms_key(
            subject_id="sub_001",
            key_id="key_001",
            kms_callback=None,
        )

        assert result is True
        mock_audit_sink.record_event.assert_called_once()

    def test_destroy_kms_key_exception(self, executor, mock_audit_sink):
        """Test KMS key destruction with exception."""
        mock_callback = Mock(side_effect=Exception("KMS error"))

        result = executor.destroy_kms_key(
            subject_id="sub_001",
            key_id="key_001",
            kms_callback=mock_callback,
        )

        assert result is False


class TestRetentionJobScheduler:
    """Test RetentionJobScheduler class."""

    @pytest.fixture
    def scheduler(self):
        """Create test scheduler."""
        config = RetentionConfig()
        return RetentionJobScheduler(config)

    def test_register_data_subject(self, scheduler):
        """Test registering a data subject."""
        record = scheduler.register_data_subject(
            subject_id="sub_001",
            subject_name="Test User",
            data_categories=["experiment", "config"],
            retention_policy=RetentionPolicy.GDPR_DEFAULT,
        )

        assert record.subject_id == "sub_001"
        assert record.subject_name == "Test User"
        assert record.data_categories == ["experiment", "config"]
        assert "sub_001" in scheduler.data_subjects

    def test_request_deletion_success(self, scheduler):
        """Test successful deletion request."""
        scheduler.register_data_subject(
            subject_id="sub_001",
            subject_name="Test User",
            data_categories=["experiment"],
        )

        result = scheduler.request_deletion("sub_001")

        assert result is True
        assert scheduler.data_subjects["sub_001"].deletion_requested is True

    def test_request_deletion_not_found(self, scheduler):
        """Test deletion request for non-existent subject."""
        result = scheduler.request_deletion("nonexistent")

        assert result is False

    def test_execute_retention_jobs_no_subjects(self, scheduler):
        """Test executing retention jobs with no subjects."""
        results = scheduler.execute_retention_jobs()

        assert results["total_subjects"] == 0
        assert results["expired_subjects"] == 0
        assert results["deletion_requested"] == 0
        assert results["deletions_completed"] == 0
        assert results["deletions_failed"] == 0

    def test_execute_retention_jobs_with_deletion_request(self, scheduler):
        """Test executing retention jobs with deletion request."""
        scheduler.register_data_subject(
            subject_id="sub_001",
            subject_name="Test User",
            data_categories=["config"],
        )
        scheduler.request_deletion("sub_001")

        with patch.object(
            scheduler.deletion_executor, "delete_config_data", return_value=True
        ):
            results = scheduler.execute_retention_jobs()

        assert results["total_subjects"] == 1
        assert results["deletion_requested"] == 1

    def test_execute_retention_jobs_with_expired_data(self, scheduler):
        """Test executing retention jobs with expired data."""
        # Create subject with old creation date
        record = DataSubjectRecord(
            subject_id="sub_001",
            subject_name="Test User",
            data_categories=["config"],
            retention_policy=RetentionPolicy.GDPR_DEFAULT,
            created_at=datetime.utcnow() - timedelta(days=2000),
        )
        scheduler.data_subjects["sub_001"] = record

        with patch.object(
            scheduler.deletion_executor, "delete_config_data", return_value=True
        ):
            results = scheduler.execute_retention_jobs()

        assert results["expired_subjects"] == 1

    def test_execute_retention_jobs_auto_delete_disabled(self, scheduler):
        """Test retention jobs with auto-delete disabled."""
        scheduler.config.auto_delete_enabled = False

        record = DataSubjectRecord(
            subject_id="sub_001",
            subject_name="Test User",
            data_categories=["config"],
            retention_policy=RetentionPolicy.GDPR_DEFAULT,
            created_at=datetime.utcnow() - timedelta(days=2000),
        )
        scheduler.data_subjects["sub_001"] = record

        results = scheduler.execute_retention_jobs()

        assert results["expired_subjects"] == 0

    def test_execute_retention_jobs_deletion_failure(self, scheduler):
        """Test retention jobs with failed deletion."""
        scheduler.register_data_subject(
            subject_id="sub_001",
            subject_name="Test User",
            data_categories=["config"],
        )
        scheduler.request_deletion("sub_001")

        with patch.object(
            scheduler.deletion_executor, "delete_config_data", return_value=False
        ):
            results = scheduler.execute_retention_jobs()

        assert results["deletions_failed"] == 1

    def test_execute_subject_deletion_all_categories(self, scheduler):
        """Test subject deletion with all data categories."""
        scheduler.register_data_subject(
            subject_id="sub_001",
            subject_name="Test User",
            data_categories=["experiment", "config", "kms_key"],
        )

        with patch.object(
            scheduler.deletion_executor, "delete_experiment_data", return_value=True
        ):
            with patch.object(
                scheduler.deletion_executor, "delete_config_data", return_value=True
            ):
                with patch.object(
                    scheduler.deletion_executor, "destroy_kms_key", return_value=True
                ):
                    result = scheduler._execute_subject_deletion(
                        scheduler.data_subjects["sub_001"]
                    )

        assert result is True
        assert scheduler.data_subjects["sub_001"].deletion_completed is True

    def test_execute_subject_deletion_exception(self, scheduler):
        """Test subject deletion with exception."""
        scheduler.register_data_subject(
            subject_id="sub_001",
            subject_name="Test User",
            data_categories=["experiment"],
        )

        with patch.object(
            scheduler.deletion_executor,
            "delete_experiment_data",
            side_effect=Exception("Delete error"),
        ):
            result = scheduler._execute_subject_deletion(
                scheduler.data_subjects["sub_001"]
            )

        assert result is False

    def test_export_subject_data_success(self, scheduler, tmp_path):
        """Test successful data export."""
        scheduler.register_data_subject(
            subject_id="sub_001",
            subject_name="Test User",
            data_categories=["experiment", "config"],
        )

        export_path = tmp_path / "export.json"
        result = scheduler.export_subject_data("sub_001", str(export_path))

        assert result is True
        assert export_path.exists()

        with open(export_path) as f:
            data = json.load(f)
            assert data["subject_id"] == "sub_001"
            assert data["subject_name"] == "Test User"

    def test_export_subject_data_not_found(self, scheduler, tmp_path):
        """Test export for non-existent subject."""
        export_path = tmp_path / "export.json"
        result = scheduler.export_subject_data("nonexistent", str(export_path))

        assert result is False

    def test_export_subject_data_failure(self, scheduler, tmp_path):
        """Test failed data export."""
        scheduler.register_data_subject(
            subject_id="sub_001",
            subject_name="Test User",
            data_categories=["experiment"],
        )

        # Use invalid path
        result = scheduler.export_subject_data(
            "sub_001", "/nonexistent/directory/export.json"
        )

        assert result is False

    def test_get_retention_statistics(self, scheduler):
        """Test getting retention statistics."""
        # Add subjects in various states
        scheduler.register_data_subject(
            subject_id="sub_001",
            subject_name="Test User 1",
            data_categories=["experiment"],
        )

        expired_record = DataSubjectRecord(
            subject_id="sub_002",
            subject_name="Test User 2",
            data_categories=["experiment"],
            retention_policy=RetentionPolicy.GDPR_DEFAULT,
            created_at=datetime.utcnow() - timedelta(days=2000),
        )
        scheduler.data_subjects["sub_002"] = expired_record

        scheduler.register_data_subject(
            subject_id="sub_003",
            subject_name="Test User 3",
            data_categories=["config"],
        )
        scheduler.request_deletion("sub_003")

        stats = scheduler.get_retention_statistics()

        assert stats["total_subjects"] == 3
        assert stats["expired_records"] == 1
        assert stats["deletion_requested"] == 1
        assert stats["pending_deletion"] == 1


class TestGlobalFunctions:
    """Test global functions."""

    def test_get_retention_scheduler(self):
        """Test getting global retention scheduler."""
        scheduler = get_retention_scheduler()
        assert isinstance(scheduler, RetentionJobScheduler)

    def test_set_retention_scheduler(self):
        """Test setting global retention scheduler."""
        new_scheduler = RetentionJobScheduler(RetentionConfig())
        set_retention_scheduler(new_scheduler)

        assert get_retention_scheduler() is new_scheduler

        # Reset to original
        set_retention_scheduler(_retention_scheduler)
