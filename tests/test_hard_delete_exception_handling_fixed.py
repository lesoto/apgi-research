"""
Test to verify the fix for test_hard_delete_exception_handling.
"""

import os
import sys
from unittest.mock import patch

import pytest

# Add parent directory to the path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.apgi_compliance import ComplianceManager


class TestHardDeleteExceptionHandlingFixed:
    """Test to verify the fix for hard delete exception handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ComplianceManager()

    def test_hard_delete_exception_handling_fixed(self):
        """Test hard delete exception handling (lines 165-167) - FIXED VERSION."""
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
                # This should trigger exception handling at lines 165-167
                with pytest.raises(OSError):
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
