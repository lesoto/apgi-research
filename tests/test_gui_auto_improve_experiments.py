"""
Test suite for GUI-auto_improve_experiments.py module.

Tests GUI functionality for APGI experiment management.
"""

import os
import pytest
import importlib
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

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

# Mock GUI dependencies after importing
sys.modules["customtkinter"] = MagicMock()
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.figure"] = MagicMock()
sys.modules["matplotlib.backends.backend_tkagg"] = MagicMock()


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

                    assert instance.title() == "APGI Auto-Improvement Research Hub"
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
        """Test menu bar creation."""
        mock_gui = MagicMock()

        with patch("customtkinter.CTkFrame") as mock_frame:
            with patch("customtkinter.CTkButton") as mock_button:
                with patch("customtkinter.CTkFont"):
                    gui.ExperimentRunnerGUI._create_menu_bar(mock_gui)

                    # Should create menu frame with buttons
                    assert mock_frame.call_count >= 1
                    assert mock_button.call_count >= 2

    def test_setup_ui(self):
        """Test UI setup."""
        mock_gui = MagicMock()
        mock_gui.experiments = ["test1", "test2"]
        mock_gui.experiment_cards = {}
        mock_gui.experiment_buttons = {}
        mock_gui.status_indicators = {}

        with patch("customtkinter.CTkScrollableFrame"):
            with patch("customtkinter.CTkLabel"):
                with patch("customtkinter.CTkButton"):
                    with patch.object(mock_gui, "_create_experiment_card"):
                        gui.ExperimentRunnerGUI._setup_ui(mock_gui)

                        # Should create experiment cards for each experiment
                        assert mock_gui._create_experiment_card.call_count == 2

    def test_create_experiment_card(self):
        """Test creating individual experiment cards."""
        mock_gui = MagicMock()
        mock_gui.experiment_cards = {}
        mock_gui.experiment_buttons = {}
        mock_gui.status_indicators = {}

        with patch("customtkinter.CTkFrame") as mock_frame:
            with patch("customtkinter.CTkLabel"):
                with patch("customtkinter.CTkButton"):
                    with patch("customtkinter.CTkProgressBar"):
                        gui.ExperimentRunnerGUI._create_experiment_card(
                            mock_gui, "test_experiment"
                        )

                        # Should create card with buttons and progress bar
                        assert mock_frame.call_count >= 1
                        # Check that progress bar was created by examining method calls
                        method_calls = str(mock_gui.method_calls)
                        assert (
                            "progress_bar" in method_calls
                            or "progressBar" in method_calls
                        )

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

        with patch("gui.subprocess.Popen") as mock_popen:
            with patch("gui.threading.Thread") as mock_thread:
                mock_process = MagicMock()
                mock_popen.return_value = mock_process
                mock_thread.return_value = MagicMock()

                gui.ExperimentRunnerGUI._run_experiment(mock_gui, "test")

                # Should update state and start process
                assert "test" in mock_gui.running_experiments
                assert mock_gui.active_processes["test"] == mock_process
                mock_button.configure.assert_called()
                mock_thread.assert_called_once()

    def test_stop_experiment(self):
        """Test stopping an experiment."""
        mock_gui = MagicMock()
        mock_gui.running_experiments = {"test"}
        mock_gui.active_processes = {}
        mock_gui.experiment_buttons = {}
        mock_gui.status_indicators = {}

        # Mock process and button
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.terminate.return_value = None
        mock_gui.active_processes["test"] = mock_process

        mock_button = MagicMock()
        mock_status = MagicMock()
        mock_gui.experiment_buttons["test"] = mock_button
        mock_gui.status_indicators["test"] = mock_status

        gui.ExperimentRunnerGUI._stop_experiment(mock_gui, "test")

        # Should terminate process and update state
        mock_process.terminate.assert_called_once()
        assert "test" not in mock_gui.running_experiments
        mock_button.configure.assert_called()

    def test_stop_experiment_already_finished(self):
        """Test stopping experiment that's already finished."""
        mock_gui = MagicMock()
        mock_gui.running_experiments = {"test"}
        mock_gui.active_processes = {}
        mock_gui.experiment_buttons = {}
        mock_gui.status_indicators = {}

        # Mock finished process
        mock_process = MagicMock()
        mock_process.poll.return_value = 0  # Already finished
        mock_gui.active_processes["test"] = mock_process

        mock_button = MagicMock()
        mock_status = MagicMock()
        mock_gui.experiment_buttons["test"] = mock_button
        mock_gui.status_indicators["test"] = mock_status

        gui.ExperimentRunnerGUI._stop_experiment(mock_gui, "test")

        # Should not call terminate on finished process
        mock_process.terminate.assert_not_called()

    def test_stop_all_experiments(self):
        """Test stopping all running experiments."""
        mock_gui = MagicMock()
        mock_gui.running_experiments = {"test1", "test2"}
        mock_gui.stop_all = False

        with patch.object(mock_gui, "_stop_experiment") as mock_stop:
            gui.ExperimentRunnerGUI._stop_all_experiments(mock_gui)

            # Should stop all experiments
            assert mock_stop.call_count == 2
            assert mock_gui.stop_all is True

    def test_monitor_experiment_output(self):
        """Test monitoring experiment output."""
        mock_gui = MagicMock()
        mock_gui.stop_all = False
        mock_gui.experiment_results = {}

        # Mock process that outputs data
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.stdout.readline.side_effect = [
            "line 1\n",
            "line 2\n",
            "",  # EOF
        ]

        with patch("gui.time.sleep"):
            gui.ExperimentRunnerGUI._monitor_experiment_output(
                mock_gui, "test", mock_process
            )

            # Should read lines and update results
            assert mock_gui.experiment_results["test"]["output"] == ["line 1", "line 2"]

    def test_monitor_experiment_output_with_stop(self):
        """Test monitoring experiment output with stop signal."""
        mock_gui = MagicMock()
        mock_gui.stop_all = True
        mock_gui.experiment_results = {}

        mock_process = MagicMock()
        mock_process.poll.return_value = None

        with patch("gui.time.sleep"):
            gui.ExperimentRunnerGUI._monitor_experiment_output(
                mock_gui, "test", mock_process
            )

            # Should terminate process when stop_all is True
            mock_process.terminate.assert_called_once()

    def test_update_experiment_status(self):
        """Test updating experiment status."""
        mock_gui = MagicMock()
        mock_gui.status_indicators = {}

        mock_status = MagicMock()
        mock_gui.status_indicators["test"] = mock_status

        gui.ExperimentRunnerGUI._update_experiment_status(
            mock_gui, "test", "Running", "green"
        )

        mock_status.configure.assert_called_once_with(
            text="Running", text_color="green"
        )

    def test_create_visualization(self):
        """Test creating visualization."""
        mock_gui = MagicMock()
        mock_gui.current_figure = None
        mock_gui.current_canvas = None

        with patch("matplotlib.figure.Figure") as mock_figure_class:
            with patch(
                "matplotlib.backends.backend_tkagg.FigureCanvasTkAgg"
            ) as mock_canvas_class:
                mock_figure = MagicMock()
                mock_canvas = MagicMock()
                mock_figure_class.return_value = mock_figure
                mock_canvas_class.return_value = mock_canvas

                data = {"accuracy": [0.8, 0.85, 0.9]}

                gui.ExperimentRunnerGUI._create_visualization(mock_gui, data)

                assert mock_gui.current_figure == mock_figure
                assert mock_gui.current_canvas == mock_canvas

    def test_clear_visualization(self):
        """Test clearing visualization."""
        mock_gui = MagicMock()
        mock_gui.current_figure = MagicMock()
        mock_gui.current_canvas = MagicMock()

        gui.ExperimentRunnerGUI._clear_visualization(mock_gui)

        mock_gui.current_figure.clear.assert_called_once()
        mock_gui.current_canvas.draw.assert_called_once()
        assert mock_gui.current_figure is None
        assert mock_gui.current_canvas is None

    def test_export_results(self):
        """Test exporting results."""
        mock_gui = MagicMock()
        mock_gui.experiment_results = {"test": {"accuracy": 0.8, "output": ["line 1"]}}

        with patch("gui.messagebox.asksaveasfilename") as mock_save:
            with patch("builtins.open", mock_open()):
                with patch("json.dump") as mock_dump:
                    mock_save.return_value = "/tmp/results.json"

                    gui.ExperimentRunnerGUI._export_results(mock_gui)

                    mock_dump.assert_called_once()

    def test_export_results_cancelled(self):
        """Test exporting results when cancelled."""
        mock_gui = MagicMock()
        mock_gui.experiment_results = {}

        with patch("gui.messagebox.asksaveasfilename") as mock_save:
            mock_save.return_value = ""  # User cancelled

            gui.ExperimentRunnerGUI._export_results(mock_gui)

            # Should not attempt to save

    def test_load_experiment_config(self):
        """Test loading experiment configuration."""
        mock_gui = MagicMock()
        mock_gui.research_dir = Path("/tmp")

        with patch("gui.importlib.util.spec_from_file_location") as mock_spec:
            with patch("gui.importlib.util.module_from_spec") as mock_module:
                with patch("gui.importlib.util.spec_loader.exec_module") as mock_exec:
                    mock_spec.return_value = MagicMock()
                    mock_module.return_value = MagicMock()

                    gui.ExperimentRunnerGUI._load_experiment_config(mock_gui, "test")

                    mock_spec.assert_called_once()
                    mock_exec.assert_called_once()

    def test_load_experiment_config_error(self):
        """Test loading experiment config with error."""
        mock_gui = MagicMock()
        mock_gui.research_dir = Path("/tmp")

        with patch("gui.importlib.util.spec_from_file_location") as mock_spec:
            mock_spec.side_effect = ImportError("Module not found")

            config = gui.ExperimentRunnerGUI._load_experiment_config(
                mock_gui, "nonexistent"
            )

            assert config is None

    def test_validate_experiment_file(self):
        """Test validating experiment file."""
        mock_gui = MagicMock()
        mock_gui.research_dir = Path("/tmp")

        # Create a valid experiment file
        mock_gui.research_dir / "run_test.py"

        with patch("gui.Path.exists") as mock_exists:
            with patch("gui.Path.is_file") as mock_is_file:
                mock_exists.return_value = True
                mock_is_file.return_value = True

                result = gui.ExperimentRunnerGUI._validate_experiment_file(
                    mock_gui, "test"
                )

                assert result is True

    def test_validate_experiment_file_not_exists(self):
        """Test validating non-existent experiment file."""
        mock_gui = MagicMock()
        mock_gui.research_dir = Path("/tmp")

        with patch("gui.Path.exists") as mock_exists:
            mock_exists.return_value = False

            result = gui.ExperimentRunnerGUI._validate_experiment_file(
                mock_gui, "nonexistent"
            )

            assert result is False

    def test_get_experiment_info(self):
        """Test getting experiment information."""
        mock_gui = MagicMock()
        mock_gui.research_dir = Path("/tmp")

        with patch.object(mock_gui, "_load_experiment_config") as mock_load:
            mock_config = MagicMock()
            mock_config.__doc__ = "Test experiment description"
            mock_load.return_value = mock_config

            info = gui.ExperimentRunnerGUI._get_experiment_info(mock_gui, "test")

            assert "description" in info
            assert "config" in info

    def test_cleanup_on_close(self):
        """Test cleanup when GUI is closed."""
        mock_gui = MagicMock()
        mock_gui.running_experiments = {"test1", "test2"}
        mock_gui.active_processes = {}

        # Mock processes
        mock_process1 = MagicMock()
        mock_process2 = MagicMock()
        mock_gui.active_processes = {"test1": mock_process1, "test2": mock_process2}

        with patch.object(mock_gui, "_stop_all_experiments") as mock_stop_all:
            gui.ExperimentRunnerGUI._cleanup_on_close(mock_gui)

            mock_stop_all.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
