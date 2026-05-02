"""
Tests for experiments/migrate_prepare_files.py - migration script for prepare files.
"""

from pathlib import Path
from unittest.mock import patch

from experiments.migrate_prepare_files import main, migrate_file


class TestMigrateFile:
    """Tests for migrate_file function."""

    def test_already_migrated(self, tmp_path):
        """Test file with cli_entrypoint already present is skipped."""
        test_file = tmp_path / "prepare_test.py"
        test_file.write_text(
            '"""Test file with cli_entrypoint."""\nimport numpy as np\nfrom apgi_cli import cli_entrypoint\n'
        )

        result = migrate_file(test_file)
        assert result is False

    def test_migrate_success(self, tmp_path):
        """Test successful migration of prepare file."""
        test_file = tmp_path / "prepare_test_experiment.py"
        test_file.write_text(
            '"""Test prepare file."""\nimport numpy as np\n\ndef verify():\n    pass\n\nif __name__ == "__main__":\n    verify()\n'
        )

        result = migrate_file(test_file)
        assert result is True

        # Verify migration applied
        content = test_file.read_text()
        assert "cli_entrypoint" in content
        assert "create_standard_parser" in content
        assert "def verify() -> int:" in content
        assert "return 0" in content
        assert "def main() -> int:" in content

    def test_migrate_without_numpy_import(self, tmp_path):
        """Test migration when numpy import not present."""
        test_file = tmp_path / "prepare_test.py"
        test_file.write_text(
            '"""Test prepare file."""\n\ndef verify():\n    pass\n\nif __name__ == "__main__":\n    verify()\n'
        )

        result = migrate_file(test_file)
        # Should still migrate but without adding numpy import
        assert result is True
        content = test_file.read_text()
        assert "cli_entrypoint" in content

    def test_migrate_preserves_existing_return(self, tmp_path):
        """Test migration doesn't duplicate return 0 if already present."""
        test_file = tmp_path / "prepare_test.py"
        test_file.write_text(
            '"""Test prepare file."""\nimport numpy as np\n\ndef verify():\n    return 0\n\nif __name__ == "__main__":\n    verify()\n'
        )

        result = migrate_file(test_file)
        assert result is True
        content = test_file.read_text()
        # Should only have one return 0
        assert content.count("return 0") == 1

    def test_migrate_complex_verify(self, tmp_path):
        """Test migration with more complex verify function."""
        test_file = tmp_path / "prepare_complex.py"
        test_file.write_text(
            '"""Complex prepare file."""\nimport numpy as np\n\ndef verify():\n    x = 1\n    y = 2\n    return x + y\n\nif __name__ == "__main__":\n    result = verify()\n    print(result)\n'
        )

        result = migrate_file(test_file)
        assert result is True
        content = test_file.read_text()
        assert "cli_entrypoint" in content
        assert "return 0" in content  # Should add return 0 after the body


class TestMain:
    """Tests for main function."""

    @patch("experiments.migrate_prepare_files.Path")
    def test_main_no_files(self, mock_path_class, tmp_path, capsys):
        """Test main when no prepare files exist."""
        mock_path = tmp_path
        mock_path_class.return_value = mock_path

        with patch(
            "experiments.migrate_prepare_files.migrate_file", return_value=False
        ):
            with patch("pathlib.Path.glob", return_value=[]):
                result = main()
                assert result == 0

    def test_main_with_files(self, tmp_path, capsys):
        """Test main with existing prepare files."""
        # Create a prepare file
        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()
        prepare_file = experiments_dir / "prepare_test.py"
        prepare_file.write_text(
            '"""Test."""\nimport numpy as np\n\ndef verify():\n    pass\n\nif __name__ == "__main__":\n    verify()\n'
        )

        with patch.object(Path, "parent", tmp_path):
            with patch.object(Path, "glob", return_value=[prepare_file]):
                main()
                captured = capsys.readouterr()
                assert "Migrated" in captured.out or "Skipped" in captured.out
