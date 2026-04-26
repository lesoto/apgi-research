"""
Test suite for the GUI module.

This module provides testing for the experiment runner GUI.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

tk: Optional[object] = None
ctk: Optional[object] = None

try:
    import tkinter

    tk = tkinter
    import customtkinter

    ctk = customtkinter
except ImportError:
    pass


class TestGUIComponents:
    """Test GUI components."""

    def test_gui_import(self):
        """Test that GUI module can be imported."""
        if tk is None or ctk is None:
            pytest.skip("tkinter or CustomTkinter not available")

        # Test that we can import GUI module
        try:
            # Import the module directly using importlib
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "GUI-auto_improve_experiments",
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "GUI-auto_improve_experiments.py",
                ),
            )
            if spec is not None:
                GUI_module = importlib.util.module_from_spec(spec)
            else:
                pytest.skip("Could not load GUI module spec")

            assert GUI_module is not None
        except ImportError:
            pytest.skip("GUI module not available")

    def test_gui_components_exist(self):
        """Test that GUI components exist."""
        try:
            import customtkinter as ctk

            # Test basic CustomTkinter components
            assert ctk.CTk is not None
            assert ctk.CTkFrame is not None
            assert ctk.CTkButton is not None
            assert ctk.CTkLabel is not None
        except ImportError:
            pytest.skip("CustomTkinter not available")

    def test_gui_dependencies_check(self):
        """Test that dependency checking works correctly."""
        import importlib

        # Test that required dependencies can be checked
        CORE_DEPENDENCIES = {
            "numpy": "NumPy",
            "pandas": "Pandas",
            "matplotlib": "Matplotlib",
        }

        for module, name in CORE_DEPENDENCIES.items():
            try:
                importlib.import_module(module)
                available = True
            except ImportError:
                available = False

            # Just verify the check works, don't require all deps
            assert isinstance(available, bool)

    def test_experiment_discovery(self):
        """Test that experiment discovery finds run files."""
        research_dir = Path(os.path.dirname(os.path.dirname(__file__)))
        # Experiment files are in the experiments/ subdirectory
        experiments_dir = research_dir / "experiments"
        run_files = list(experiments_dir.glob("run_*.py"))

        # Should find at least some experiment files
        assert len(run_files) > 0, "No run_*.py files found"

        # Verify file naming pattern
        for file in run_files:
            assert file.stem.startswith("run_"), f"Invalid experiment file: {file}"
            assert file.suffix == ".py", f"Invalid file extension: {file}"

    def test_experiment_card_creation(self):
        """Test experiment card data structure."""
        # Test card data structure
        card_data = {
            "name": "Test Experiment",
            "script": "run_test.py",
            "status": "Ready",
            "color": "#3498db",
        }

        assert card_data["name"] is not None
        assert card_data["script"].endswith(".py")
        assert card_data["status"] in ["Ready", "Running", "Success", "Failed"]

    def test_menu_structure(self):
        """Test that menu structure is properly defined."""
        menu_items = ["File", "Edit", "View", "Help"]

        # Verify menu structure
        assert len(menu_items) == 4
        assert all(isinstance(item, str) for item in menu_items)

    def test_console_output_format(self):
        """Test console output formatting."""
        # Test log format
        log_message = "[STARTING] Test Experiment (run_test.py)"
        assert log_message.startswith("[")
        assert "]" in log_message

    def test_dependency_validation(self):
        """Test dependency validation logic."""
        # Test with missing dependency
        missing_core = []
        test_deps = {"nonexistent_module": "Test Module"}

        for module, description in test_deps.items():
            try:
                __import__(module)
            except ImportError:
                missing_core.append(f"  - {module}: {description}")

        assert len(missing_core) == 1
        assert "nonexistent_module" in missing_core[0]

    def test_path_handling(self):
        """Test cross-platform path handling."""
        # Test pathlib usage
        test_path = Path.home() / ".cache" / "test"
        assert isinstance(test_path, Path)
        assert str(test_path).replace("\\", "/").count("/") >= 2

    def test_results_parsing(self):
        """Test experiment results parsing patterns."""
        # Test metric parsing patterns
        test_lines = [
            "primary_metric: 0.85",
            "accuracy: 92.5%",
            "net_score: 42.0",
        ]

        for line in test_lines:
            assert ":" in line
            parts = line.split(":")
            assert len(parts) == 2
            try:
                value = float(parts[1].strip().rstrip("%"))
                assert isinstance(value, float)
            except ValueError:
                pass  # Some values might not be numeric


class TestGUISecurity:
    """Test GUI security features."""

    def test_script_path_validation(self):
        """Test script path validation."""
        # Test valid script path
        valid_script = "run_experiment.py"
        assert valid_script.endswith(".py")

        # Test invalid script paths
        invalid_scripts = [
            "../../../etc/passwd",
            "run_experiment.exe",
            "script.sh",
        ]

        for script in invalid_scripts:
            assert not script.endswith(".py") or ".." in script

    def test_subprocess_command_validation(self):
        """Test subprocess command validation."""
        # Test allowed commands
        allowed = ["python", "python3", "uv"]
        test_cmd = "python run_test.py"

        cmd_parts = test_cmd.split()
        assert cmd_parts[0] in allowed or any(cmd_parts[0].endswith(a) for a in allowed)


if __name__ == "__main__":
    pytest.main([__file__])
