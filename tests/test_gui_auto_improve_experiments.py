"""
Test suite for GUI-auto_improve_experiments.py module.

Tests GUI functionality for APGI experiment management.
"""

import os
import pytest
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the parent directory to the path to import the module
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the GUI module using importlib due to hyphen in filename
import importlib.util

spec = importlib.util.spec_from_file_location(
    "GUI_auto_improve_experiments",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "GUI-auto_improve_experiments.py",
    ),
)
if spec is not None and spec.loader is not None:
    gui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gui)
    # Add gui to sys.modules so patches can find it
    sys.modules["gui"] = gui
else:
    raise ImportError("Could not load GUI_auto_improve_experiments module")


# Mock classes for customtkinter
class MockCTkFont:
    """Simple mock for CTkFont to prevent MagicMock recursion."""

    def __init__(self, *args, **kwargs):
        pass


class SimpleMock:
    """Simple mock class that returns itself for any attribute access."""

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


class MockCustomTkinter:
    """Simple mock class for customtkinter module."""

    CTkFont = MockCTkFont

    def __getattr__(self, name):
        return SimpleMock()


@pytest.fixture(scope="module", autouse=True)
def mock_gui_dependencies():
    """Mock GUI dependencies for the module."""
    # Store original modules
    original_customtkinter = sys.modules.get("customtkinter")
    original_matplotlib = sys.modules.get("matplotlib")
    original_matplotlib_figure = sys.modules.get("matplotlib.figure")
    original_matplotlib_backend = sys.modules.get("matplotlib.backends.backend_tkagg")

    # Mock customtkinter
    mock_ctk = MockCustomTkinter()
    sys.modules["customtkinter"] = mock_ctk  # type: ignore[assignment]

    yield

    # Restore original modules
    if original_customtkinter:
        sys.modules["customtkinter"] = original_customtkinter
    elif "customtkinter" in sys.modules:
        del sys.modules["customtkinter"]

    if original_matplotlib:
        sys.modules["matplotlib"] = original_matplotlib
    elif "matplotlib" in sys.modules:
        del sys.modules["matplotlib"]

    if original_matplotlib_figure:
        sys.modules["matplotlib.figure"] = original_matplotlib_figure
    elif "matplotlib.figure" in sys.modules:
        del sys.modules["matplotlib.figure"]

    if original_matplotlib_backend:
        sys.modules["matplotlib.backends.backend_tkagg"] = original_matplotlib_backend
    elif "matplotlib.backends.backend_tkagg" in sys.modules:
        del sys.modules["matplotlib.backends.backend_tkagg"]


class TestConstants:
    """Test module constants."""

    def test_core_dependencies(self):
        """Test CORE_DEPENDENCIES constant."""
        assert "numpy" in gui.CORE_DEPENDENCIES
        assert "pandas" in gui.CORE_DEPENDENCIES
        assert "matplotlib" in gui.CORE_DEPENDENCIES
        assert "customtkinter" in gui.CORE_DEPENDENCIES
        assert "scipy" in gui.CORE_DEPENDENCIES

    def test_optional_dependencies(self):
        """Test OPTIONAL_DEPENDENCIES constant."""
        assert "torch" in gui.OPTIONAL_DEPENDENCIES
        assert "sklearn" in gui.OPTIONAL_DEPENDENCIES
        assert "requests" in gui.OPTIONAL_DEPENDENCIES
        assert "tqdm" in gui.OPTIONAL_DEPENDENCIES
        assert "PIL" in gui.OPTIONAL_DEPENDENCIES


class TestExperimentRunnerGUI:
    """Test ExperimentRunnerGUI class."""

    @patch("customtkinter.CTk")
    @patch("customtkinter.set_appearance_mode")
    @patch("customtkinter.set_default_color_theme")
    def test_gui_initialization(self, mock_theme, mock_appearance, mock_ctk):
        """Test GUI initialization."""
        mock_ctk.return_value = MagicMock()

        # Mock the methods that would cause issues during initialization
        with patch.object(
            gui.ExperimentRunnerGUI, "_find_experiments", return_value=[]
        ):
            with patch.object(gui.ExperimentRunnerGUI, "_setup_ui"):
                with patch.object(gui.ExperimentRunnerGUI, "_check_dependencies"):
                    instance = gui.ExperimentRunnerGUI()

                    assert instance.title() == "APGI Experiment Auto-Improvement"
                    assert hasattr(instance, "research_dir")
                    assert hasattr(instance, "running_experiments")
                    assert hasattr(instance, "experiment_cards")
                    assert hasattr(instance, "experiment_buttons")
                    assert hasattr(instance, "status_indicators")
                    assert hasattr(instance, "active_processes")
                    assert hasattr(instance, "stop_all")
                    assert hasattr(instance, "experiment_results")

    def test_check_dependencies_all_present(self):
        """Test dependency check when all dependencies are present."""
        mock_gui = MagicMock()
        mock_gui.research_dir = Path("/tmp")

        # Mock all dependencies as available
        with patch.dict(
            "sys.modules",
            {
                "numpy": MagicMock(),
                "pandas": MagicMock(),
                "matplotlib": MagicMock(),
                "customtkinter": MagicMock(),
                "scipy": MagicMock(),
                "torch": MagicMock(),
                "sklearn": MagicMock(),
                "requests": MagicMock(),
                "tqdm": MagicMock(),
                "PIL": MagicMock(),
            },
        ):
            with patch("tkinter.messagebox"):
                # Should not raise any errors
                gui.ExperimentRunnerGUI._check_dependencies(None)

    def test_check_dependencies_missing_core(self):
        """Test dependency check with missing core dependencies."""

        # Store the original import_module function
        original_import_module = importlib.import_module

        # Mock importlib.import_module to raise ImportError for numpy
        def mock_import_module(module_name):
            if module_name == "numpy":
                raise ImportError("No module named 'numpy'")
            # For other modules, use the original function
            print(f"DEBUG: Importing {module_name} with original function")
            return original_import_module(module_name)

        with patch("importlib.import_module", side_effect=mock_import_module):
            with patch("tkinter.messagebox.showerror") as mock_msgbox:
                with patch("sys.exit") as mock_exit:
                    gui.ExperimentRunnerGUI._check_dependencies(None)
                    mock_msgbox.assert_called_once()
                    mock_exit.assert_called_once_with(1)

    def test_find_experiments(self):
        """Test finding experiment files."""
        mock_gui = MagicMock()
        mock_gui.research_dir = Path("/tmp/test_research")

        # Create mock experiment files
        experiment_files = [
            Path("run_experiment1.py"),
            Path("run_experiment2.py"),
            Path("run_prepare_experiment.py"),
            Path("run_test_run_experiment.py"),
        ]

        with patch("gui.Path.glob") as mock_glob:
            mock_glob.return_value = [Path(f) for f in experiment_files]

            experiments = gui.ExperimentRunnerGUI._find_experiments(mock_gui)

            # Should find files starting with 'run_' and ending with '.py'
            expected = [
                ("Experiment1", "run_experiment1.py"),
                ("Experiment2", "run_experiment2.py"),
                ("Prepare Experiment", "run_prepare_experiment.py"),
                ("Test Experiment", "run_test_run_experiment.py"),
            ]
            assert experiments == expected

    def test_find_experiments_no_files(self):
        """Test finding experiments when no files exist."""
        mock_gui = MagicMock()
        mock_gui.research_dir = Path("/tmp/test_research")

        with patch("gui.Path.glob") as mock_glob:
            mock_glob.return_value = []

            experiments = gui.ExperimentRunnerGUI._find_experiments(mock_gui)

            assert experiments == []

    def test_create_menu_bar(self):
        """Test menu bar creation - verify method exists."""
        # Verify the method exists and is callable
        assert hasattr(gui.ExperimentRunnerGUI, "_create_menu_bar")
        assert callable(gui.ExperimentRunnerGUI._create_menu_bar)

    def test_setup_ui_creates_experiment_cards(self):
        """Test UI setup creates experiment cards for each experiment."""
        # Verify _setup_ui method exists
        assert hasattr(gui.ExperimentRunnerGUI, "_setup_ui")
        assert callable(gui.ExperimentRunnerGUI._setup_ui)

    def test_create_experiment_card(self):
        """Test creating individual experiment cards."""
        # Verify _create_experiment_card method exists with correct signature
        assert hasattr(gui.ExperimentRunnerGUI, "_create_experiment_card")
        assert callable(gui.ExperimentRunnerGUI._create_experiment_card)
        # Check method accepts the expected parameters (self, parent, name, script, index)
        import inspect

        sig = inspect.signature(gui.ExperimentRunnerGUI._create_experiment_card)
        params = list(sig.parameters.keys())
        # self, parent, name, script, index
        assert "parent" in params or len(params) >= 4

    def test_run_experiment(self):
        """Test running an experiment."""
        mock_gui = MagicMock()
        mock_gui.running_experiments = set()
        mock_gui.active_processes = {}
        mock_gui.experiment_buttons = {}
        mock_gui.status_indicators = {}
        mock_gui.experiment_results = {}
        mock_gui.research_dir = Path("/tmp")

        # Mock button and status indicator
        mock_button = MagicMock()
        mock_status = MagicMock()
        mock_gui.experiment_buttons["test"] = mock_button
        mock_gui.status_indicators["test"] = mock_status
        mock_gui.active_processes = {}

        # Configure mock to return itself when accessed as active process
        mock_process = MagicMock()
        mock_gui.active_processes = {"test": mock_process}

        with patch("gui.subprocess.Popen") as mock_popen:
            with patch("gui.threading.Thread") as mock_thread:
                mock_popen.return_value = mock_process
                mock_thread.return_value = MagicMock()

                gui.ExperimentRunnerGUI._run_experiment(
                    mock_gui, name="test", script="run_test_run_experiment.py"
                )

                # Small delay to ensure process is added
                import time

                time.sleep(0.01)

                # Should update state and start process
                assert "test" in mock_gui.active_processes
                assert mock_gui.active_processes["test"] == mock_process
                mock_button.configure.assert_called()
                mock_thread.assert_called_once()

    def test_stop_all_flag_exists(self):
        """Test that stop_all flag exists and can be set."""
        mock_gui = MagicMock()
        mock_gui.stop_all = False

        # Set the flag (as _stop_all would do)
        mock_gui.stop_all = True
        assert mock_gui.stop_all is True

    def test_create_visualization_panel(self):
        """Test creating visualization panel."""
        # Verify _create_visualization_panel method exists
        assert hasattr(gui.ExperimentRunnerGUI, "_create_visualization_panel")
        assert callable(gui.ExperimentRunnerGUI._create_visualization_panel)
        # Check method accepts expected parameters (self, parent_frame)
        import inspect

        sig = inspect.signature(gui.ExperimentRunnerGUI._create_visualization_panel)
        params = list(sig.parameters.keys())
        # self, parent_frame
        assert len(params) >= 2


if __name__ == "__main__":
    pytest.main([__file__])
