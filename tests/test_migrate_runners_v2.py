"""
Tests for experiments/migrate_runners_v2.py - simplified migration script.
"""

from pathlib import Path
from unittest.mock import patch

from experiments.migrate_runners_v2 import (
    RUNNER_FILES,
    add_standardized_imports,
    main,
    migrate_file,
    update_main_entrypoint,
)


class TestAddStandardizedImports:
    """Tests for add_standardized_imports function."""

    def test_already_has_imports(self):
        """Test when imports already exist."""
        content = """
import numpy as np
from standard_apgi_runner import StandardAPGIRunner
from apgi_cli import cli_entrypoint
"""
        result = add_standardized_imports(content)
        assert result == content

    def test_adds_imports_correctly(self):
        """Test adding standardized imports."""
        content = """
import numpy as np
import pandas as pd

print("test")
"""
        result = add_standardized_imports(content)
        assert "StandardAPGIRunner" in result
        assert "cli_entrypoint" in result
        assert "create_standard_parser" in result
        assert "# Standardized APGI imports" in result

    def test_finds_last_import(self):
        """Test that imports are added after the last import."""
        content = """
import sys
import os

x = 1
"""
        result = add_standardized_imports(content)
        lines = result.split("\n")
        import_idx = [
            i
            for i, line in enumerate(lines)
            if line.startswith("import") or line.startswith("from ")
        ]
        std_idx = [
            i for i, line in enumerate(lines) if "Standardized APGI imports" in line
        ][0]
        assert std_idx > max(import_idx)


class TestUpdateMainEntrypoint:
    """Tests for update_main_entrypoint function."""

    def test_removes_existing_main(self):
        """Test removal of existing __main__ block."""
        content = """
def run():
    pass

if __name__ == "__main__":
    run()
"""
        result = update_main_entrypoint(content, "run_test.py")
        assert "cli_entrypoint" in result
        assert 'if __name__ == "__main__":' in result

    def test_creates_main_function(self):
        """Test creation of main function with args parameter."""
        content = ""
        result = update_main_entrypoint(content, "run_stroop_effect.py")
        assert "def main(args):" in result
        assert "EnhancedStroopEffectRunner" in result

    def test_proper_class_naming(self):
        """Test proper class name generation from filename."""
        content = ""
        result = update_main_entrypoint(content, "run_ai_benchmarking.py")
        assert "EnhancedAiBenchmarkingRunner" in result
        result2 = update_main_entrypoint(content, "run_working_memory_span.py")
        assert "EnhancedWorkingMemorySpanRunner" in result2

    def test_removes_simple_main_function(self):
        """Test removal of simple main() at end of file."""
        content = """
class Runner:
    pass

def main():
    return results
"""
        result = update_main_entrypoint(content, "run_test.py")
        # Old main should be removed and replaced
        assert result.count("def main(") == 1


class TestMigrateFile:
    """Tests for migrate_file function."""

    def test_successful_migration(self, tmp_path):
        """Test successful migration."""
        test_file = tmp_path / "run_test.py"
        test_file.write_text("""
import numpy as np

class TestRunner:
    def run_experiment(self):
        return {}

if __name__ == "__main__":
    runner = TestRunner()
    runner.run_experiment()
""")
        result = migrate_file(test_file)
        assert result is True

        content = test_file.read_text()
        assert "StandardAPGIRunner" in content
        assert "cli_entrypoint" in content

    def test_no_changes_needed(self, tmp_path):
        """Test when file already has standardized imports."""
        test_file = tmp_path / "run_test.py"
        test_file.write_text("""
from standard_apgi_runner import StandardAPGIRunner
from apgi_cli import cli_entrypoint

if __name__ == "__main__":
    cli_entrypoint(main, parser)
""")
        result = migrate_file(test_file)
        assert result is False

    def test_error_handling(self, tmp_path):
        """Test error handling during migration."""
        test_file = tmp_path / "run_test.py"
        test_file.write_text("test content")

        with patch("builtins.open", side_effect=IOError("Test error")):
            result = migrate_file(test_file)
            assert result is False


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
                assert "Migrating" in captured.out
                assert "Migrated:" in captured.out

    def test_main_output_summary(self, capsys, tmp_path):
        """Test main function output format."""
        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()
        runner_file = experiments_dir / "run_test.py"
        runner_file.write_text("""
class TestRunner:
    def run_experiment(self):
        return {}
""")

        with patch("experiments.migrate_runners_v2.RUNNER_FILES", ["run_test.py"]):
            with patch.object(Path, "parent", experiments_dir):
                with patch.object(Path, "__truediv__", return_value=runner_file):
                    with patch.object(Path, "exists", return_value=True):
                        main()
                        captured = capsys.readouterr()
                        assert "Migrating" in captured.out
                        assert "=" in captured.out
                        assert "Migrated:" in captured.out
                        assert "Skipped:" in captured.out
                        assert "Total:" in captured.out
