"""
Test suite for delete_pycache.py module.

Tests file and directory cleanup functionality with comprehensive coverage.
"""

import errno
import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add the parent directory to the path to import the module
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import delete_pycache as dp


class TestUtilityFunctions:
    """Test utility functions."""

    def test_matches_any_positive(self):
        """Test matches_any with matching patterns."""
        assert dp.matches_any("test.py", ["*.py", "*.txt"])
        assert dp.matches_any("__pycache__", ["__pycache__", "*.egg-info"])
        assert dp.matches_any("debug_test.py", ["debug_*.py"])

    def test_matches_any_negative(self):
        """Test matches_any with non-matching patterns."""
        assert not dp.matches_any("test.py", ["*.txt", "*.md"])
        assert not dp.matches_any("normal_file", ["debug_*", "temp_*"])

    def test_matches_any_empty_patterns(self):
        """Test matches_any with empty patterns."""
        assert not dp.matches_any("anything", [])


class TestDirectoryRemoval:
    """Test directory removal logic."""

    def test_should_remove_directory_default_names(self):
        """Test directory removal with default names."""
        assert dp._should_remove_directory(
            "__pycache__",
            dp.DEFAULT_DIR_NAMES,
            dp.DEFAULT_DIR_PATTERNS,
            [],
            False,
            False,
            [],
        )
        assert dp._should_remove_directory(
            ".pytest_cache",
            dp.DEFAULT_DIR_NAMES,
            dp.DEFAULT_DIR_PATTERNS,
            [],
            False,
            False,
            [],
        )

    def test_should_remove_directory_patterns(self):
        """Test directory removal with patterns."""
        assert dp._should_remove_directory(
            "test.egg-info",
            dp.DEFAULT_DIR_NAMES,
            dp.DEFAULT_DIR_PATTERNS,
            [],
            False,
            False,
            [],
        )

    def test_should_remove_directory_custom_patterns(self):
        """Test directory removal with custom patterns."""
        assert dp._should_remove_directory(
            "custom_dir", set(), [], ["custom_*"], False, False, []
        )

    def test_should_remove_directory_node_modules(self):
        """Test node_modules removal."""
        assert dp._should_remove_directory(
            "node_modules", set(), [], [], True, False, []
        )
        assert not dp._should_remove_directory(
            "node_modules", set(), [], [], False, False, []
        )

    def test_should_remove_directory_venvs(self):
        """Test virtual environment removal."""
        assert dp._should_remove_directory(
            ".venv", set(), [], [], False, True, [".venv", "venv"]
        )
        assert not dp._should_remove_directory(
            ".venv", set(), [], [], False, True, ["env"]
        )

    def test_remove_directory_dry_run(self):
        """Test directory removal in dry run mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_dir"
            test_dir.mkdir()

            stats = {"dirs_removed": 0, "errors": 0}
            dp._remove_directory(
                temp_dir, "test_dir", dry_run=True, verbose=False, stats=stats
            )
            assert stats["dirs_removed"] == 0
            assert test_dir.exists()

    def test_remove_directory_actual_removal(self):
        """Test actual directory removal."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_dir"
            test_dir.mkdir()

            stats = {"dirs_removed": 0, "errors": 0}
            dp._remove_directory(
                temp_dir, "test_dir", dry_run=False, verbose=False, stats=stats
            )
            assert stats["dirs_removed"] == 1
            assert not test_dir.exists()

    def test_remove_directory_with_retry(self):
        """Test directory removal with retry logic."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_dir"
            test_dir.mkdir()

            # Mock shutil.rmtree to fail first then succeed
            with patch("shutil.rmtree") as mock_rmtree:
                mock_rmtree.side_effect = [
                    OSError(errno.EBUSY, "Busy"),  # First call fails
                    None,  # Second call succeeds
                ]

                stats = {"dirs_removed": 0, "errors": 0}
                dp._remove_directory(
                    temp_dir,
                    "test_dir",
                    dry_run=False,
                    verbose=False,
                    stats=stats,
                    max_retries=2,
                )
                assert stats["dirs_removed"] == 1
                assert mock_rmtree.call_count == 2

    def test_remove_directory_max_retries_exceeded(self):
        """Test directory removal when max retries exceeded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_dir"
            test_dir.mkdir()

            with patch("shutil.rmtree") as mock_rmtree:
                mock_rmtree.side_effect = OSError(errno.EBUSY, "Busy")

                stats = {"dirs_removed": 0, "errors": 0}
                dp._remove_directory(
                    temp_dir,
                    "test_dir",
                    dry_run=False,
                    verbose=False,
                    stats=stats,
                    max_retries=2,
                )
                assert stats["dirs_removed"] == 0
                assert stats["errors"] == 1


class TestFileRemoval:
    """Test file removal logic."""

    def test_remove_file_dry_run(self):
        """Test file removal in dry run mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test content")

            stats = {"files_removed": 0, "errors": 0}
            dp._remove_file(str(test_file), dry_run=True, verbose=False, stats=stats)
            assert stats["files_removed"] == 0
            assert test_file.exists()

    def test_remove_file_actual_removal(self):
        """Test actual file removal."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test content")

            stats = {"files_removed": 0, "errors": 0}
            dp._remove_file(str(test_file), dry_run=False, verbose=False, stats=stats)
            assert stats["files_removed"] == 1
            assert not test_file.exists()

    def test_remove_file_with_retry(self):
        """Test file removal with retry logic."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test content")

            with patch("os.remove") as mock_remove:
                mock_remove.side_effect = [
                    OSError(errno.EACCES, "Permission denied"),  # First call fails
                    None,  # Second call succeeds
                ]

                stats = {"files_removed": 0, "errors": 0}
                dp._remove_file(
                    str(test_file),
                    dry_run=False,
                    verbose=False,
                    stats=stats,
                    max_retries=2,
                )
                assert stats["files_removed"] == 1
                assert mock_remove.call_count == 2

    def test_remove_file_not_found(self):
        """Test file removal when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "nonexistent.txt"

            stats = {"files_removed": 0, "errors": 0}
            dp._remove_file(str(test_file), dry_run=False, verbose=False, stats=stats)
            # File not found should be counted as success (ENOENT handling)
            assert stats["files_removed"] == 1


class TestDirectoryProcessing:
    """Test directory processing logic."""

    def test_process_directories_removal(self):
        """Test processing directories for removal."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test directories
            pycache_dir = Path(temp_dir) / "__pycache__"
            normal_dir = Path(temp_dir) / "normal"
            pycache_dir.mkdir()
            normal_dir.mkdir()

            dirnames = ["__pycache__", "normal"]
            stats = {"dirs_removed": 0, "errors": 0}

            dp._process_directories(
                temp_dir,
                dirnames,
                dp.DEFAULT_DIR_NAMES,
                dp.DEFAULT_DIR_PATTERNS,
                [],
                [],
                False,
                False,
                [],
                dry_run=False,
                verbose=False,
                stats=stats,
            )

            assert stats["dirs_removed"] == 1
            assert "__pycache__" not in dirnames
            assert "normal" in dirnames

    def test_process_directories_exclude_patterns(self):
        """Test processing directories with exclude patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pycache_dir = Path(temp_dir) / "__pycache__"
            pycache_dir.mkdir()

            dirnames = ["__pycache__"]
            stats = {"dirs_removed": 0, "errors": 0}

            dp._process_directories(
                temp_dir,
                dirnames,
                dp.DEFAULT_DIR_NAMES,
                dp.DEFAULT_DIR_PATTERNS,
                [],
                ["__pycache__"],  # Exclude __pycache__
                False,
                False,
                [],
                dry_run=False,
                verbose=False,
                stats=stats,
            )

            assert stats["dirs_removed"] == 0


class TestFileProcessing:
    """Test file processing logic."""

    def test_process_files_removal(self):
        """Test processing files for removal."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            pyc_file = Path(temp_dir) / "test.pyc"
            normal_file = Path(temp_dir) / "test.txt"
            pyc_file.write_text("bytecode")
            normal_file.write_text("content")

            filenames = ["test.pyc", "test.txt"]
            stats = {"files_removed": 0, "errors": 0}

            dp._process_files(
                temp_dir,
                filenames,
                dp.DEFAULT_FILE_PATTERNS,
                [],
                [],
                dry_run=False,
                verbose=False,
                stats=stats,
            )

            assert stats["files_removed"] == 1
            assert not pyc_file.exists()
            assert normal_file.exists()

    def test_process_files_exclude_patterns(self):
        """Test processing files with exclude patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pyc_file = Path(temp_dir) / "test.pyc"
            pyc_file.write_text("bytecode")

            filenames = ["test.pyc"]
            stats = {"files_removed": 0, "errors": 0}

            dp._process_files(
                temp_dir,
                filenames,
                dp.DEFAULT_FILE_PATTERNS,
                [],
                ["*.pyc"],  # Exclude .pyc files
                dry_run=False,
                verbose=False,
                stats=stats,
            )

            assert stats["files_removed"] == 0


class TestDepthControl:
    """Test directory depth control."""

    def test_should_skip_directory_no_limit(self):
        """Test depth skipping with no limit."""
        assert not dp._should_skip_directory("/a/b/c", "/a", None)

    def test_should_skip_directory_within_limit(self):
        """Test depth skipping within limit."""
        assert not dp._should_skip_directory("/a/b", "/a", 2)
        assert not dp._should_skip_directory("/a/b/c", "/a", 3)

    def test_should_skip_directory_exceeds_limit(self):
        """Test depth skipping when exceeds limit."""
        assert dp._should_skip_directory("/a/b/c/d", "/a", 3) is False
        assert dp._should_skip_directory("/a/b/c/d/e", "/a", 3) is True


class TestPreviewFunctionality:
    """Test preview functionality."""

    def test_preview_deletions_basic(self):
        """Test basic preview functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test structure
            pycache_dir = Path(temp_dir) / "__pycache__"
            pyc_file = Path(temp_dir) / "test.pyc"
            pycache_dir.mkdir()
            pyc_file.write_text("bytecode")

            stats = dp.preview_deletions(temp_dir)

            assert len(stats["dirs_to_remove"]) == 1
            assert len(stats["files_to_remove"]) == 1
            assert stats["total_files"] >= 1
            assert stats["total_size_bytes"] > 0
            assert stats["errors"] == 0

    def test_preview_deletions_with_patterns(self):
        """Test preview with custom patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test structure
            custom_dir = Path(temp_dir) / "custom_temp"
            custom_file = Path(temp_dir) / "custom.tmp"
            custom_dir.mkdir()
            custom_file.write_text("content")

            stats = dp.preview_deletions(
                temp_dir,
                include_dir_patterns=["custom_*"],
                include_file_patterns=["*.tmp"],
            )

            assert len(stats["dirs_to_remove"]) == 1
            assert len(stats["files_to_remove"]) == 1

    def test_preview_deletions_exclude_patterns(self):
        """Test preview with exclude patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test structure
            pycache_dir = Path(temp_dir) / "__pycache__"
            pyc_file = Path(temp_dir) / "test.pyc"
            pycache_dir.mkdir()
            pyc_file.write_text("bytecode")

            stats = dp.preview_deletions(
                temp_dir,
                exclude_dir_patterns=["__pycache__"],
                exclude_file_patterns=["*.pyc"],
            )

            assert len(stats["dirs_to_remove"]) == 0
            assert len(stats["files_to_remove"]) == 0


class TestMainFunctions:
    """Test main cleanup functions."""

    def test_delete_temporary_items_basic(self):
        """Test basic temporary item deletion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test structure
            pycache_dir = Path(temp_dir) / "__pycache__"
            pyc_file = Path(temp_dir) / "test.pyc"
            pycache_dir.mkdir()
            pyc_file.write_text("bytecode")

            stats = dp.delete_temporary_items(temp_dir, dry_run=False, verbose=False)

            assert stats["dirs_removed"] >= 1
            assert stats["files_removed"] >= 1

    def test_delete_temporary_items_dry_run(self):
        """Test temporary item deletion in dry run mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test structure
            pycache_dir = Path(temp_dir) / "__pycache__"
            pyc_file = Path(temp_dir) / "test.pyc"
            pycache_dir.mkdir()
            pyc_file.write_text("bytecode")

            stats = dp.delete_temporary_items(temp_dir, dry_run=True, verbose=False)

            assert stats["dirs_removed"] == 0
            assert stats["files_removed"] == 0
            assert pycache_dir.exists()
            assert pyc_file.exists()

    def test_prune_empty_dirs(self):
        """Test empty directory pruning."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested empty directories
            empty_dir = Path(temp_dir) / "empty" / "nested" / "dir"
            empty_dir.mkdir(parents=True)

            # Create a file in another directory to keep it
            keep_dir = Path(temp_dir) / "keep"
            keep_dir.mkdir()
            (keep_dir / "file.txt").write_text("content")

            dp.prune_empty_dirs(temp_dir, dry_run=False, verbose=False)

            assert not empty_dir.exists()
            assert keep_dir.exists()

    def test_clear_log_files_truncate(self):
        """Test log file truncation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir) / "logs"
            logs_dir.mkdir()
            log_file = logs_dir / "test.log"
            log_file.write_text("log content")

            dp.clear_log_files(
                temp_dir, delete_logs_dir=False, dry_run=False, verbose=False
            )

            assert log_file.exists()
            assert log_file.stat().st_size == 0

    def test_clear_log_files_delete_dir(self):
        """Test log directory deletion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir) / "logs"
            logs_dir.mkdir()
            log_file = logs_dir / "test.log"
            log_file.write_text("log content")

            dp.clear_log_files(
                temp_dir, delete_logs_dir=True, dry_run=False, verbose=False
            )

            assert not logs_dir.exists()


class TestArgumentParsing:
    """Test command line argument parsing."""

    def test_parse_args_defaults(self):
        """Test default argument parsing."""
        args = dp.parse_args([])
        assert args.root is None
        assert args.dry_run is False
        assert args.preview is False
        assert args.yes is False
        assert args.quiet is False

    def test_parse_args_custom_values(self):
        """Test custom argument parsing."""
        args = dp.parse_args(
            [
                "/path/to/root",
                "--dry-run",
                "--yes",
                "--quiet",
                "--delete-logs",
                "--remove-node-modules",
                "--remove-venvs",
                "--apgi-only",
                "--keep-visualizations",
                "--keep-reports",
                "--prune-empty-dirs",
            ]
        )
        assert args.root == "/path/to/root"
        assert args.dry_run is True
        assert args.yes is True
        assert args.quiet is True
        assert args.delete_logs is True
        assert args.remove_node_modules is True
        assert args.remove_venvs is True
        assert args.apgi_only is True
        assert args.keep_visualizations is True
        assert args.keep_reports is True
        assert args.prune_empty_dirs is True

    def test_parse_args_include_exclude_patterns(self):
        """Test include/exclude pattern arguments."""
        args = dp.parse_args(
            [
                "--include-dir",
                "custom_*",
                "--include-dir",
                "temp_*",
                "--include-file",
                "*.tmp",
                "--exclude-dir",
                "important_*",
                "--exclude-file",
                "*.keep",
            ]
        )
        assert args.include_dir == ["custom_*", "temp_*"]
        assert args.include_file == ["*.tmp"]
        assert args.exclude_dir == ["important_*"]
        assert args.exclude_file == ["*.keep"]


class TestMainFunction:
    """Test main function execution."""

    def test_main_dry_run(self):
        """Test main function with dry run."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test structure
            pycache_dir = Path(temp_dir) / "__pycache__"
            pycache_dir.mkdir()

            result = dp.main([temp_dir, "--dry-run", "--yes"])
            assert result == 0

    def test_main_preview(self):
        """Test main function with preview."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = dp.main([temp_dir, "--preview"])
            assert result == 0

    def test_main_invalid_directory(self):
        """Test main function with invalid directory."""
        result = dp.main(["/nonexistent/directory", "--yes"])
        assert result == 1

    def test_main_apgi_only_mode(self):
        """Test main function in APGI-only mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create APGI-specific files
            apgi_dir = Path(temp_dir) / "apgi_output"
            debug_file = Path(temp_dir) / "debug_test.py"
            apgi_dir.mkdir()
            debug_file.write_text("debug content")

            result = dp.main([temp_dir, "--apgi-only", "--yes"])
            assert result == 0

    def test_main_keep_visualizations(self):
        """Test main function keeping visualizations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create visualization files
            png_file = Path(temp_dir) / "plot.png"
            html_file = Path(temp_dir) / "report.html"
            pyc_file = Path(temp_dir) / "test.pyc"
            png_file.write_bytes(b"png data")
            html_file.write_text("<html>report</html>")
            pyc_file.write_bytes(b"bytecode")

            result = dp.main([temp_dir, "--keep-visualizations", "--yes"])
            assert result == 0
            assert png_file.exists()  # Should be kept
            assert html_file.exists()  # Should be kept
            assert not pyc_file.exists()  # Should be removed


if __name__ == "__main__":
    pytest.main([__file__])
