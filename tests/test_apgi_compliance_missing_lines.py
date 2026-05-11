"""
Focused test suite to cover missing lines in apgi_compliance.py.
"""

import os
import sys
import tempfile
from unittest.mock import mock_open, patch

import pytest

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.apgi_compliance import ComplianceManager


class TestMissingLinesCoverage:
    """Test cases specifically targeting uncovered lines."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ComplianceManager()

    def test_hard_delete_exception_handling(self):
        """Test hard delete exception handling (lines 165-167)."""
        record = {
            "id": "test_record",
            "data": "test data",
            "created_at": "2023-01-01T00:00:00Z",
        }

        # Mock glob to return file paths
        with patch("glob.glob") as mock_glob:
            mock_glob.return_value = ["/nonexistent/file.txt"]

            # Mock os.remove to raise an exception
            with patch("os.remove", side_effect=OSError("Permission denied")):
                # This should trigger the exception handling at lines 165-167
                # The current implementation catches and logs but doesn't re-raise
                # So we need to check that the exception was handled
                self.manager._execute_deletion(record, "hard_delete")

                # Verify the exception was handled by checking audit trail
                exception_handled = False
                for entry in self.manager.audit_trail:
                    if (
                        entry.get("action") == "hard_delete"
                        and entry.get("record_id") == record["id"]
                        and "error" in entry.get("details", {})
                    ):
                        exception_handled = True
                        break

                assert (
                    exception_handled
                ), "Exception should have been handled in audit trail"

    def test_secure_erase_file_operations(self):
        """Test secure erase file operations (lines 247-258)."""
        record = {
            "id": "test_record",
            "sensitive_data": "very sensitive",
            "created_at": "2023-01-01T00:00:00Z",
        }

        # Create a temporary file for testing
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(b"original content")
        temp_file.close()

        try:
            # Mock glob to return our temp file
            with patch("glob.glob") as mock_glob:
                mock_glob.return_value = [temp_file.name]

                # Mock os.path.getsize to return a size
                with patch("os.path.getsize", return_value=16):
                    # Mock open to capture the secure erase operations
                    with patch(
                        "builtins.open", return_value=mock_open().return_value
                    ) as mock_file:
                        with patch("os.fsync"):
                            with patch("os.remove"):
                                self.manager._execute_deletion(record, "secure_erase")

                                # Verify multiple overwrite passes were attempted
                                handle = mock_file
                                handle = mock_file()
                                # Should have 3 overwrite passes + 1 zero-fill = 4 total writes
                                assert (
                                    handle.write.call_count == 4
                                )  # nosec: B101 - Test assertion

                                # Check that random data was written (3 times)
                                random_writes = [
                                    call
                                    for call in handle.write.call_args_list
                                    if call[0][0] != b"\x00" * 16
                                ]  # Not the zero-fill
                                assert (
                                    len(random_writes) == 3
                                )  # nosec: B101 - Test assertion

                                # Check final zero-fill
                                zero_fill_call = handle.write.call_args_list[-1]
                                assert (
                                    zero_fill_call[0][0] == b"\x00" * 16
                                )  # nosec: B101 - Test assertion
        finally:
            # Clean up temp file
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    def test_secure_erase_with_secrets_module(self):
        """Test secure erase uses secrets module for random data (lines 248)."""
        record = {
            "id": "test_record",
            "sensitive_data": "very sensitive",
            "created_at": "2023-01-01T00:00:00Z",
        }

        # Mock secrets module
        with patch("secrets.token_bytes") as mock_token_bytes:
            mock_token_bytes.return_value = b"random_data"

            with patch("glob.glob") as mock_glob:
                mock_glob.return_value = ["/test/file.txt"]

                with patch("os.path.getsize", return_value=10):
                    with patch("builtins.open", mock_open()):
                        with patch("os.fsync"):
                            with patch("os.remove"):
                                self.manager._execute_deletion(record, "secure_erase")

                                # Verify secrets.token_bytes was called for random data
                                assert (  # nosec: B101 - Test assertion
                                    mock_token_bytes.call_count == 3
                                )  # 3 overwrite passes (3 for random data + 1 for zero-fill)

    def test_anonymous_deletion_with_dict_record(self):
        """Test anonymous deletion with dict record (lines 264-266)."""
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
            "created_at": "2023-01-01T00:00:00Z",
        }

        self.manager._execute_deletion(record, "anonymous")

        # Check PII fields were hashed
        assert record["user_id"].startswith("hashed_")  # nosec: B101 - Test assertion
        assert record["email"].startswith("hashed_")  # nosec: B101 - Test assertion
        assert record["name"].startswith("hashed_")  # nosec: B101 - Test assertion
        assert record["ip_address"].startswith(
            "hashed_"
        )  # nosec: B101 - Test assertion

        # Check direct identifiers were removed
        assert "session_id" not in record  # nosec: B101 - Test assertion
        assert "token" not in record  # nosec: B101 - Test assertion
        assert "api_key" not in record  # nosec: B101 - Test assertion

        # Check regular field was preserved
        assert (
            record["regular_field"] == "regular_value"
        )  # nosec: B101 - Test assertion

    def test_archive_deletion_with_data_store_cleanup(self):
        """Test archive deletion with data store cleanup (lines 298, 300)."""
        record = {
            "id": "test_record",
            "data": "test data to archive",
            "created_at": "2023-01-01T00:00:00Z",
        }

        # Test deletion execution
        with patch("os.makedirs"):
            with patch("gzip.open", mock_open()) as mock_gzip:
                with patch("json.dump") as mock_json:
                    self.manager._execute_deletion(record, "archive")

                    # Check archive was created
                    mock_gzip.assert_called_once()
                    mock_json.assert_called_once()

    def test_archive_deletion_file_error_handling(self):
        """Test archive deletion with file error handling (lines 306)."""
        record = {
            "id": "test_record",
            "data": "test data",
            "created_at": "2023-01-01T00:00:00Z",
        }

        # Mock file operations to raise exception
        with patch("os.makedirs", side_effect=OSError("Permission denied")):
            # Should raise exception for file operation errors
            with pytest.raises(OSError):
                self.manager._execute_deletion(record, "archive")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
