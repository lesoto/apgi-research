"""
Tests for experiments/migrate_runners.py - migration script for runner files.
"""

from pathlib import Path
from unittest.mock import patch

from experiments.migrate_runners import (
    RUNNER_FILES,
    extract_experiment_name,
    main,
    migrate_file,
    update_imports,
    update_main_entrypoint,
    update_runner_class,
)


class TestExtractExperimentName:
    """Tests for extract_experiment_name function."""

    def test_standard_name(self):
        """Test extracting experiment name from standard filename."""
        result = extract_experiment_name("run_stroop_effect.py")
        assert result == "stroop_effect"

    def test_complex_name(self):
        """Test extracting experiment name with multiple underscores."""
        result = extract_experiment_name("run_change_blindness_full_apgi.py")
        assert result == "change_blindness_full_apgi"


class TestUpdateImports:
    """Tests for update_imports function."""

    def test_remove_old_imports(self):
        """Test removal of old APGI imports."""
        content = """
from apgi_integration import APGIIntegration, APGIParameters
from apgi_integration import format_apgi_output
print("test")
"""
        result = update_imports(content, "test_experiment")
        assert "Removed: migrated" in result
        assert "from apgi_integration import APGIIntegration" not in result

    def test_add_standardized_imports(self):
        """Test addition of standardized imports."""
        content = """
import numpy as np
print("test")
"""
        result = update_imports(content, "test_experiment")
        assert "StandardAPGIRunner" in result
        assert "cli_entrypoint" in result
        assert "create_standard_parser" in result

    def test_preserve_existing_imports(self):
        """Test that existing imports are preserved."""
        content = """
import numpy as np
import pandas as pd
print("test")
"""
        result = update_imports(content, "test_experiment")
        assert "import numpy as np" in result
        assert "import pandas as pd" in result


class TestUpdateRunnerClass:
    """Tests for update_runner_class function."""

    def test_no_runner_class(self):
        """Test handling when no runner class exists."""
        content = """
def some_function():
    pass
"""
        result = update_runner_class(content, "test_experiment")
        assert result == content

    def test_adds_apgi_runner(self):
        """Test adding StandardAPGIRunner to existing class."""
        content = """
class TestRunner:
    def __init__(self):
        self.x = 1
"""
        result = update_runner_class(content, "test_experiment")
        assert "StandardAPGIRunner" in result
        assert 'experiment_name="test_experiment"' in result

    def test_preserves_existing_apgi(self):
        """Test that existing APGI integration is not duplicated."""
        content = """
class TestRunner:
    def __init__(self):
        self.apgi_runner = StandardAPGIRunner()
"""
        result = update_runner_class(content, "test_experiment")
        # Should not duplicate
        assert result.count("StandardAPGIRunner(") == 1


class TestUpdateMainEntrypoint:
    """Tests for update_main_entrypoint function."""

    def test_replaces_main_block(self):
        """Test replacement of existing __main__ block."""
        content = """
def main():
    pass

if __name__ == "__main__":
    main()
"""
        result = update_main_entrypoint(content, "stroop_effect")
        assert "cli_entrypoint(main, parser)" in result
        assert "create_standard_parser" in result

    def test_creates_proper_class_name(self):
        """Test that class name is properly formatted."""
        content = ""
        result = update_main_entrypoint(content, "stroop_effect")
        assert "EnhancedStroopEffectRunner" in result

    def test_handles_multi_word_names(self):
        """Test handling of multi-word experiment names."""
        content = ""
        result = update_main_entrypoint(content, "change_blindness_full_apgi")
        assert "EnhancedChangeBlindnessFullApgiRunner" in result


class TestMigrateFile:
    """Tests for migrate_file function."""

    def test_successful_migration(self, tmp_path):
        """Test successful file migration."""
        test_file = tmp_path / "run_test.py"
        test_file.write_text("""
class TestRunner:
    def __init__(self):
        pass

def main():
    runner = TestRunner()
    return {}

if __name__ == "__main__":
    main()
""")
        success, message = migrate_file(test_file, "test_experiment")
        assert success is True
        assert "Successfully migrated" in message

    def test_no_changes_needed(self, tmp_path):
        """Test when no changes are needed."""
        test_file = tmp_path / "run_test.py"
        # File already migrated
        test_file.write_text("""
from standard_apgi_runner import StandardAPGIRunner
from apgi_cli import cli_entrypoint

def main():
    pass

if __name__ == "__main__":
    cli_entrypoint(main, parser)
""")
        success, message = migrate_file(test_file, "test_experiment")
        assert success is False
        assert "No changes needed" in message

    def test_error_handling(self, tmp_path):
        """Test error handling during migration."""
        test_file = tmp_path / "run_test.py"
        # Create a file but make it unreadable or cause another error
        test_file.write_text("test content")

        with patch("builtins.open", side_effect=IOError("Test error")):
            success, message = migrate_file(test_file, "test_experiment")
            assert success is False
            assert "Error" in message


class TestRunnerFilesList:
    """Tests for RUNNER_FILES constant."""

    def test_contains_expected_files(self):
        """Test that RUNNER_FILES contains expected experiment files."""
        assert "run_stroop_effect.py" in RUNNER_FILES
        assert "run_ai_benchmarking.py" in RUNNER_FILES
        assert "run_attentional_blink.py" in RUNNER_FILES
        assert len(RUNNER_FILES) == 28


class TestMain:
    """Tests for main function."""

    def test_main_with_missing_files(self, capsys):
        """Test main function when files don't exist."""
        with patch.object(Path, "exists", return_value=False):
            with patch.object(Path, "parent", Path("/fake/path")):
                main()
                captured = capsys.readouterr()
                assert "SKIP" in captured.out or "Migration complete" in captured.out

    def test_main_summary_output(self, capsys, tmp_path):
        """Test main function prints proper summary."""
        # Create a fake experiments directory with one runner file
        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()
        runner_file = experiments_dir / "run_test.py"
        runner_file.write_text("""
class TestRunner:
    def __init__(self):
        pass

if __name__ == "__main__":
    pass
""")

        with patch("experiments.migrate_runners.RUNNER_FILES", ["run_test.py"]):
            with patch.object(Path, "parent", experiments_dir):
                with patch.object(Path, "__truediv__", return_value=runner_file):
                    with patch.object(Path, "exists", return_value=True):
                        main()
                        captured = capsys.readouterr()
                        assert "Migration complete" in captured.out
                        assert (
                            "Successfully migrated" in captured.out
                            or "Skipped" in captured.out
                        )
