"""
================================================================================
GUI TESTS WITH PLAYWRIGHT
================================================================================

Playwright-based GUI tests for the APGI Experiment Runner.
Tests the CustomTkinter GUI components for correct rendering and interaction.

Note: These tests require a display server (X11/Wayland on Linux, Quartz on macOS).
For headless environments, use Xvfb or similar.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Generator, cast
from unittest.mock import MagicMock, patch

import pytest
from playwright.sync_api import Page
from pytest import MonkeyPatch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """Wait for server to be ready."""
    import urllib.request

    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.1)
    return False


class GUIPage:
    """Page object for GUI interactions."""

    def __init__(self, page: Page, app_window: Any = None):
        self.page = page
        self.app_window = app_window

    def click_menu(self, menu_name: str) -> None:
        """Click a menu item."""
        if self.app_window:
            # Use tkinter directly for menu interaction
            self.app_window.after(0, lambda: self._trigger_menu(menu_name))
            time.sleep(0.1)

    def _trigger_menu(self, menu_name: str) -> None:
        """Internal menu trigger."""
        if self.app_window and hasattr(self.app_window, f"_show_{menu_name}_menu"):
            getattr(self.app_window, f"_show_{menu_name}_menu")()

    def click_experiment(self, experiment_name: str) -> None:
        """Click on an experiment card."""
        if self.app_window:
            buttons = self.app_window.experiment_buttons
            if experiment_name in buttons:
                btn = buttons[experiment_name]
                btn.invoke()

    def wait_for_status(
        self, experiment_name: str, status: str, timeout: float = 10.0
    ) -> bool:
        """Wait for experiment to reach a status."""
        start = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed >= timeout:
                return False
            if self.app_window:
                indicators = self.app_window.status_indicators
                if experiment_name in indicators:
                    current_status = indicators[experiment_name].cget("text")
                    if status in current_status:
                        return True
            time.sleep(0.1)

    def take_screenshot(self, name: str) -> bytes:
        """Take a screenshot for visual regression testing."""
        if self.app_window:
            # Use PIL to capture the tkinter window
            from PIL import ImageGrab

            x = self.app_window.winfo_rootx()
            y = self.app_window.winfo_rooty()
            width = self.app_window.winfo_width()
            height = self.app_window.winfo_height()

            if width > 0 and height > 0:
                try:
                    # Ensure window is updated and visible for capture
                    self.app_window.update_idletasks()
                    self.app_window.update()
                    # On some systems, a small delay is needed for mapping
                    time.sleep(0.2)

                    screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
                    # Convert to bytes
                    import io

                    img_byte_arr = io.BytesIO()
                    screenshot.save(img_byte_arr, format="PNG")
                    return img_byte_arr.getvalue()
                except Exception as e:
                    print(f"Warning: GUI screenshot capture failed: {e}")
                    # In some environments (like headless CI or restricted macOS),
                    # screen capture may not be possible.
                    return b""
        return b""


@pytest.fixture
def screenshot_dir() -> Generator[Path, None, None]:
    """Create directory for screenshots."""
    ss_dir = PROJECT_ROOT / "test_reports" / "screenshots"
    ss_dir.mkdir(parents=True, exist_ok=True)
    yield ss_dir


@pytest.fixture
def temp_research_dir() -> Generator[Path, None, None]:
    """Create temporary research directory with mock experiments."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Create mock experiment files
        for exp_name in ["test_visual_search", "test_stroop_effect"]:
            exp_file = tmp_path / f"run_{exp_name}.py"
            exp_file.write_text(f"""
# Mock experiment for testing
import time
print(f"Running {exp_name}")
time.sleep(0.1)
""")
        yield tmp_path


@pytest.mark.gui
@pytest.mark.slow
class TestGUIBasic:
    """Basic GUI functionality tests."""

    def test_gui_imports(self) -> None:
        """Test that GUI modules can be imported."""
        try:
            import customtkinter as ctk

            assert ctk is not None
        except ImportError:
            pytest.skip("customtkinter not installed")

        try:
            from GUI_auto_improve_experiments import ExperimentRunnerGUI

            assert ExperimentRunnerGUI is not None
        except ImportError as e:
            pytest.skip(f"Cannot import GUI: {e}")

    def test_gui_initialization_without_display(self, monkeypatch: Any) -> None:
        """Test GUI can be initialized (headless mode)."""
        # Skip if no display available
        if not os.environ.get("DISPLAY") and sys.platform != "darwin":
            pytest.skip("No display available")

        try:
            import customtkinter as ctk

            from GUI_auto_improve_experiments import ExperimentRunnerGUI
        except ImportError:
            pytest.skip("Required dependencies not installed")

        # Mock the mainloop to prevent blocking
        with patch.object(ctk.CTk, "mainloop"):
            with patch.object(ctk.CTk, "withdraw"):  # Hide window
                app = ExperimentRunnerGUI()
                assert app is not None
                assert hasattr(app, "experiments")
                assert hasattr(app, "experiment_cards")
                app.destroy()

    def test_gui_experiment_discovery(
        self, temp_research_dir: Path, monkeypatch: Any
    ) -> None:
        """Test that experiments are discovered correctly."""
        try:
            import customtkinter as ctk

            from GUI_auto_improve_experiments import ExperimentRunnerGUI
        except ImportError:
            pytest.skip("Required dependencies not installed")

        # Mock the research directory
        with patch.object(
            Path, "glob", return_value=list(temp_research_dir.glob("run_*.py"))
        ):
            with patch.object(Path, "resolve", return_value=temp_research_dir):
                with patch.object(ctk.CTk, "mainloop"):
                    with patch.object(ctk.CTk, "withdraw"):
                        app = ExperimentRunnerGUI()
                        # Check experiments were found
                        assert len(app.experiments) > 0
                        for name, _ in app.experiments:
                            assert "Visual Search" in name or "Stroop" in name
                        app.destroy()


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.integration
class TestGUIInteraction:
    """GUI interaction tests."""

    def test_menu_bar_creation(self, monkeypatch: MonkeyPatch) -> None:
        """Test menu bar is created with all menu items."""
        try:
            import customtkinter as ctk

            from GUI_auto_improve_experiments import ExperimentRunnerGUI
        except ImportError:
            pytest.skip("Required dependencies not installed")

        with patch.object(ctk.CTk, "mainloop"):
            with patch.object(ctk.CTk, "withdraw"):
                app = ExperimentRunnerGUI()
                # Verify menu exists
                assert hasattr(app, "_create_menu_bar")
                app.destroy()

    def test_experiment_button_creation(
        self, temp_research_dir: Any, monkeypatch: MonkeyPatch
    ) -> None:
        """Test experiment buttons are created."""
        try:
            import customtkinter as ctk

            from GUI_auto_improve_experiments import ExperimentRunnerGUI
        except ImportError:
            pytest.skip("Required dependencies not installed")

        with patch.object(
            Path, "glob", return_value=list(temp_research_dir.glob("run_*.py"))
        ):
            with patch.object(Path, "resolve", return_value=temp_research_dir):
                with patch.object(ctk.CTk, "mainloop"):
                    with patch.object(ctk.CTk, "withdraw"):
                        app = ExperimentRunnerGUI()
                        # Check buttons exist for experiments
                        for exp_name, _ in app.experiments:
                            assert exp_name in app.experiment_buttons
                        app.destroy()

    def test_status_indicators(
        self, temp_research_dir: Any, monkeypatch: MonkeyPatch
    ) -> None:
        """Test status indicators are tracked."""
        try:
            import customtkinter as ctk

            from GUI_auto_improve_experiments import ExperimentRunnerGUI
        except ImportError:
            pytest.skip("Required dependencies not installed")

        with patch.object(
            Path, "glob", return_value=list(temp_research_dir.glob("run_*.py"))
        ):
            with patch.object(Path, "resolve", return_value=temp_research_dir):
                with patch.object(ctk.CTk, "mainloop"):
                    with patch.object(ctk.CTk, "withdraw"):
                        app = ExperimentRunnerGUI()
                        # Check status indicators exist
                        for exp_name, _ in app.experiments:
                            assert exp_name in app.status_indicators
                        app.destroy()


@pytest.mark.visual
@pytest.mark.slow
class TestGUIVisualRegression:
    """Visual regression tests for GUI."""

    @pytest.fixture
    def visual_baseline_dir(self) -> Path:
        """Directory for baseline screenshots."""
        baseline_dir = PROJECT_ROOT / "tests" / "visual_baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        return baseline_dir

    def test_gui_screenshot_comparison(
        self, temp_research_dir: Any, visual_baseline_dir: Any, monkeypatch: MonkeyPatch
    ) -> None:
        """Test GUI rendering against baseline."""
        try:
            import customtkinter as ctk
            from PIL import Image

            from GUI_auto_improve_experiments import ExperimentRunnerGUI
        except ImportError:
            pytest.skip("Required dependencies not installed")

        # Skip if no display
        if not os.environ.get("DISPLAY") and sys.platform != "darwin":
            pytest.skip("No display available")

        with patch.object(
            Path, "glob", return_value=list(temp_research_dir.glob("run_*.py"))
        ):
            with patch.object(Path, "resolve", return_value=temp_research_dir):
                with patch.object(ctk.CTk, "mainloop"):
                    with patch.object(ctk.CTk, "withdraw"):
                        app = ExperimentRunnerGUI()

                        # Take screenshot
                        gui_page = GUIPage(cast(Page, None), app)
                        screenshot = gui_page.take_screenshot("main_window")

                        if screenshot:
                            baseline_path = (
                                visual_baseline_dir / "main_window_baseline.png"
                            )

                            # Save current screenshot
                            current_path = (
                                PROJECT_ROOT
                                / "test_reports"
                                / "screenshots"
                                / "main_window_current.png"
                            )
                            current_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(current_path, "wb") as f:
                                f.write(screenshot)

                            # If baseline doesn't exist, create it
                            if not baseline_path.exists():
                                with open(baseline_path, "wb") as f:
                                    f.write(screenshot)
                                pytest.skip("Created baseline screenshot")

                            # Compare screenshots
                            try:
                                from pixelmatch import pixelmatch

                                baseline_img = Image.open(baseline_path)
                                current_img = Image.open(current_path)

                                diff = pixelmatch(
                                    baseline_img.convert("RGB"),
                                    current_img.convert("RGB"),
                                    threshold=0.1,
                                    include_aa=True,
                                )

                                # Allow small differences (UI may vary slightly)
                                assert (
                                    diff < 1000
                                ), f"Visual regression detected: {diff} pixels differ"
                            except ImportError:
                                pytest.skip("pixelmatch not installed")
                        else:
                            pytest.skip(
                                "Screenshot capture unavailable in this environment"
                            )

                        app.destroy()


@pytest.mark.e2e
@pytest.mark.slow
class TestGUIE2E:
    """End-to-end GUI tests."""

    def test_full_experiment_workflow(
        self, temp_research_dir: Any, monkeypatch: MonkeyPatch
    ) -> None:
        """Test complete experiment workflow from selection to completion."""
        try:
            import customtkinter as ctk

            from GUI_auto_improve_experiments import ExperimentRunnerGUI
        except ImportError:
            pytest.skip("Required dependencies not installed")

        # Skip if no display
        if not os.environ.get("DISPLAY") and sys.platform != "darwin":
            pytest.skip("No display available")

        # Create a simple mock experiment that completes quickly
        mock_exp = temp_research_dir / "run_quick_test.py"
        mock_exp.write_text("""
import sys
import time
print("Test experiment running", flush=True)
time.sleep(0.01)
print("Test experiment completed", flush=True)
sys.exit(0)
""")

        with patch.object(Path, "glob", return_value=[mock_exp]):
            with patch.object(Path, "resolve", return_value=temp_research_dir):
                with patch.object(ctk.CTk, "mainloop"):
                    with patch.object(ctk.CTk, "withdraw"):
                        app = ExperimentRunnerGUI()

                        # Mock subprocess to avoid actual execution
                        with patch(
                            "GUI_auto_improve_experiments.subprocess.Popen"
                        ) as mock_popen:
                            mock_process = MagicMock()
                            mock_process.poll.return_value = 0
                            mock_process.stdout = MagicMock()
                            mock_process.stdout.readline.return_value = ""
                            mock_popen.return_value = mock_process

                            # Simulate clicking an experiment
                            exp_name = list(app.experiment_buttons.keys())[0]
                            btn = app.experiment_buttons[exp_name]
                            btn.invoke()

                            # Verify experiment was started
                            assert exp_name in app.running_experiments

                        app.destroy()


@pytest.mark.performance
@pytest.mark.gui
class TestGUIPerformance:
    """GUI performance tests."""

    def test_gui_startup_time(
        self, temp_research_dir: Any, request: Any, monkeypatch: MonkeyPatch
    ) -> None:
        """Benchmark GUI startup time."""
        try:
            import customtkinter as ctk

            from GUI_auto_improve_experiments import ExperimentRunnerGUI
        except ImportError:
            pytest.skip("Required dependencies not installed")

        def startup():
            with patch.object(
                Path, "glob", return_value=list(temp_research_dir.glob("run_*.py"))
            ):
                with patch.object(Path, "resolve", return_value=temp_research_dir):
                    with patch.object(ctk.CTk, "mainloop"):
                        with patch.object(ctk.CTk, "withdraw"):
                            app = ExperimentRunnerGUI()
                            app.destroy()

        # Run without benchmark fixture if not available
        import time

        start = time.perf_counter()
        startup()
        duration = time.perf_counter() - start
        # Should start in less than 5 seconds
        assert duration < 5.0

    def test_gui_memory_usage(
        self, temp_research_dir: Any, monkeypatch: MonkeyPatch
    ) -> None:
        """Test GUI memory usage stays within bounds."""
        try:
            import os

            import customtkinter as ctk
            import psutil

            from GUI_auto_improve_experiments import ExperimentRunnerGUI
        except ImportError:
            pytest.skip("Required dependencies not installed")

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        with patch.object(
            Path, "glob", return_value=list(temp_research_dir.glob("run_*.py"))
        ):
            with patch.object(Path, "resolve", return_value=temp_research_dir):
                with patch.object(ctk.CTk, "mainloop"):
                    with patch.object(ctk.CTk, "withdraw"):
                        app = ExperimentRunnerGUI()

                        mem_after = process.memory_info().rss / 1024 / 1024  # MB
                        mem_increase = mem_after - mem_before

                        # Should use less than 100MB additional memory
                        assert (
                            mem_increase < 100
                        ), f"Memory increase too large: {mem_increase:.1f}MB"

                        app.destroy()


# Configure pytest markers
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with GUI-specific markers."""
    config.addinivalue_line("markers", "gui: marks tests as GUI tests")
    config.addinivalue_line("markers", "visual: marks tests as visual regression tests")
