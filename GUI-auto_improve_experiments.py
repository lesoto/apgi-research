"""APGI Experiment Runner GUI (Premium Edition)
Modernized for apgi-research directory with CustomTkinter.
"""

from tkinter import messagebox
import customtkinter as ctk  # type: ignore
import subprocess
import threading
import os
import sys
import re
import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict, Set
import importlib.util

# Matplotlib imports for embedded visualization
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Set appearance mode and color theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Setup logging
logger = logging.getLogger(__name__)

# Core dependencies required for APGI experiments
CORE_DEPENDENCIES = {
    "numpy": "NumPy - Numerical computing",
    "pandas": "Pandas - Data manipulation",
    "matplotlib": "Matplotlib - Plotting and visualization",
    "customtkinter": "CustomTkinter - Modern UI framework",
    "scipy": "SciPy - Scientific computing",
}

OPTIONAL_DEPENDENCIES = {
    "torch": "PyTorch - Deep learning framework",
    "sklearn": "Scikit-learn - Machine learning",
    "requests": "Requests - HTTP library",
    "tqdm": "tqdm - Progress bars",
    "PIL": "Pillow - Image processing",
}


class ExperimentRunnerGUI(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("APGI Auto-Improvement Research Hub")
        self.geometry("1400x900")

        # Set main path to current research directory
        self.research_dir = Path(os.path.dirname(os.path.abspath(__file__)))

        # Create menu bar
        self._create_menu_bar()

        # Application state
        self.running_experiments: Set[str] = set()
        self.experiment_cards: Dict[str, ctk.CTkFrame] = {}
        self.experiment_buttons: Dict[str, ctk.CTkButton] = {}
        self.status_indicators: Dict[str, ctk.CTkLabel] = {}
        self.active_processes: Dict[str, subprocess.Popen] = {}
        self.stop_all = False
        self.current_figure: Figure | None = None
        self.current_canvas: FigureCanvasTkAgg | None = None
        self.experiment_results: Dict[str, dict] = {}

        # Find experiments
        self.experiments = self._find_experiments()

        self._setup_ui()

        # Check dependencies after UI is initialized so we can log to console
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check for required dependencies on startup and show error if missing."""
        missing_core = []
        for module, description in CORE_DEPENDENCIES.items():
            try:
                __import__(module)
            except ImportError:
                missing_core.append(f"  - {module}: {description}")

        if missing_core:
            error_msg = "Missing required dependencies:\n\n" + "\n".join(missing_core)
            error_msg += "\n\nInstall with: pip install " + " ".join(
                mod.split("[")[0] for mod, _ in CORE_DEPENDENCIES.items()
            )
            messagebox.showerror("Missing Dependencies", error_msg)
            sys.exit(1)

        # Log optional dependencies
        missing_optional = []
        for module, description in OPTIONAL_DEPENDENCIES.items():
            try:
                __import__(module)
            except ImportError:
                missing_optional.append(f"  - {module}: {description}")

        if missing_optional:
            print("Optional dependencies missing (some features may be unavailable):")
            for msg in missing_optional:
                print(msg)

    def _find_experiments(self) -> List[Tuple[str, str]]:
        """Dynamically find all run_*.py files in the research directory."""
        experiments = []
        run_files = sorted(list(self.research_dir.glob("run_*.py")))

        for file in run_files:
            # Format name: run_visual_search.py -> Visual Search
            name = file.stem.replace("run_", "").replace("_", " ").title()
            if name == "Tests":
                continue  # Skip run_tests.py if it exists here
            experiments.append((name, file.name))

        return experiments

    def _create_menu_bar(self) -> None:
        """Create the application menu bar with interactive buttons."""
        menubar = ctk.CTkFrame(self, height=40)
        menubar.grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=(0, 0), pady=(0, 0)
        )

        # File menu button
        file_btn = ctk.CTkButton(
            menubar,
            text="File",
            font=ctk.CTkFont(size=12, weight="bold"),
            width=60,
            fg_color="transparent",
            hover_color="#333333",
            command=self._show_file_menu,
        )
        file_btn.pack(side="left", padx=5, pady=5)

        # Edit menu button
        edit_btn = ctk.CTkButton(
            menubar,
            text="Edit",
            font=ctk.CTkFont(size=12, weight="bold"),
            width=60,
            fg_color="transparent",
            hover_color="#333333",
            command=self._show_edit_menu,
        )
        edit_btn.pack(side="left", padx=5, pady=5)

        # View menu button
        view_btn = ctk.CTkButton(
            menubar,
            text="View",
            font=ctk.CTkFont(size=12, weight="bold"),
            width=60,
            fg_color="transparent",
            hover_color="#333333",
            command=self._show_view_menu,
        )
        view_btn.pack(side="left", padx=5, pady=5)

        # Help menu button
        help_btn = ctk.CTkButton(
            menubar,
            text="Help",
            font=ctk.CTkFont(size=12, weight="bold"),
            width=60,
            fg_color="transparent",
            hover_color="#333333",
            command=self._show_help_menu,
        )
        help_btn.pack(side="left", padx=5, pady=5)

        # Adjust grid layout to account for menu bar
        self.grid_rowconfigure(1, weight=3)  # Main content area
        self.grid_rowconfigure(2, weight=1)  # Console area

    def _setup_ui(self) -> None:
        # Configure grid layout (3x2: menu, sidebar+main, console)
        self.grid_columnconfigure(0, weight=0)  # Sidebar fixed width
        self.grid_columnconfigure(1, weight=1)  # Main content expandable
        self.grid_rowconfigure(0, weight=0)  # Menu bar fixed height
        self.grid_rowconfigure(1, weight=3)  # Main content area (3x weight)
        self.grid_rowconfigure(2, weight=1)  # Console area (1x weight)

        # Create navigation frame (sidebar)
        self.navigation_frame = ctk.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=1, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)

        self.navigation_frame_label = ctk.CTkLabel(
            self.navigation_frame,
            text="APGI HUB",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.run_all_button = ctk.CTkButton(
            self.navigation_frame,
            text="▶ Run All Experiments",
            command=self._run_all,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#27ae60",
            hover_color="#219150",
        )
        self.run_all_button.grid(row=1, column=0, padx=20, pady=10)

        self.stop_button = ctk.CTkButton(
            self.navigation_frame,
            text="⏹ Stop All",
            command=self._stop_all,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#e74c3c",
            hover_color="#c0392b",
        )
        self.stop_button.grid(row=2, column=0, padx=20, pady=10)

        self.clear_button = ctk.CTkButton(
            self.navigation_frame,
            text="🧹 Clear Console",
            command=self._clear_console,
            height=40,
        )
        self.clear_button.grid(row=3, column=0, padx=20, pady=10)

        self.appearance_mode_label = ctk.CTkLabel(
            self.navigation_frame, text="Appearance:", anchor="w"
        )
        self.appearance_mode_label.grid(row=4, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(
            self.navigation_frame,
            values=["Dark", "Light", "System"],
            command=self.change_appearance_mode,
        )
        self.appearance_mode_optionemenu.grid(row=5, column=0, padx=20, pady=(10, 20))

        # Create main scrollable area for experiments
        self.scrollable_frame = ctk.CTkScrollableFrame(
            self, label_text=f"Research Experiments ({len(self.experiments)})"
        )
        self.scrollable_frame.grid(
            row=1, column=1, padx=(20, 10), pady=(20, 10), sticky="nsew"
        )
        self.scrollable_frame.grid_columnconfigure((0, 1), weight=1)

        # Populate experiments
        for i, (name, script) in enumerate(self.experiments):
            self._create_experiment_card(self.scrollable_frame, name, script, i)

        # Create output console (bottom)
        self.console_frame = ctk.CTkFrame(self, height=250)
        self.console_frame.grid(
            row=2, column=0, columnspan=2, padx=20, pady=(10, 20), sticky="nsew"
        )
        self.console_frame.grid_columnconfigure(0, weight=1)
        self.console_frame.grid_rowconfigure(0, weight=1)

        self.console_text = ctk.CTkTextbox(self.console_frame, font=("Courier", 13))
        self.console_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.console_text.insert("0.0", "--- APGI Research Console Ready ---\n")

    def _create_experiment_card(self, parent, name, script, index) -> None:
        row = index // 2
        col = index % 2

        card = ctk.CTkFrame(parent)
        card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)

        name_label = ctk.CTkLabel(
            card, text=name, font=ctk.CTkFont(size=16, weight="bold")
        )
        name_label.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")

        script_label = ctk.CTkLabel(
            card, text=f"📄 {script}", font=ctk.CTkFont(size=12), text_color="gray"
        )
        script_label.grid(row=1, column=0, padx=15, pady=(0, 10), sticky="w")

        status_label = ctk.CTkLabel(
            card, text="Ready", font=ctk.CTkFont(size=12), text_color="#3498db"
        )
        status_label.grid(row=2, column=0, padx=15, pady=(0, 15), sticky="w")
        self.status_indicators[name] = status_label

        run_btn = ctk.CTkButton(
            card,
            text="▶ RUN",
            command=lambda n=name, s=script: self._run_experiment(n, s),
            width=70,
            height=28,
            fg_color="#27ae60",
            hover_color="#219150",
        )
        run_btn.grid(row=0, column=1, padx=(0, 5), pady=(15, 5))
        self.experiment_buttons[name] = run_btn

        viz_btn = ctk.CTkButton(
            card,
            text="📊 VIZ",
            command=lambda n=name: self._show_results_visualization(n),
            width=70,
            height=28,
            fg_color="#3498db",
            hover_color="#2980b9",
        )
        viz_btn.grid(row=1, column=1, padx=(0, 5), pady=(0, 15))

        self.experiment_cards[name] = card

    def _log(self, text, color=None):
        self.console_text.insert("end", text + "\n")
        self.console_text.see("end")

    def _clear_console(self):
        self.console_text.delete("1.0", "end")
        self.console_text.insert("0.0", "--- Console Cleared ---\n")

    def _run_experiment(self, name, script) -> None:
        if name in self.running_experiments:
            return

        self.running_experiments.add(name)
        self.experiment_buttons[name].configure(state="disabled")
        self.status_indicators[name].configure(text="Running...", text_color="#f39c12")
        self._log(f"\n[STARTING] {name} ({script})", "#3498db")

        thread = threading.Thread(target=self._execute_script, args=(name, script))
        thread.daemon = True
        thread.start()

    def _execute_script(self, name, script, output_callback=None):
        """Execute script with optional output callback and input sanitization."""
        try:
            # Validate and sanitize script name
            if not script.endswith(".py"):
                raise ValueError("Only Python scripts (.py) are allowed")

            # Validate script path is within research directory
            script_path = Path(self.research_dir) / script
            if not script_path.exists():
                raise ValueError(f"Script not found: {script}")

            # Ensure script is within allowed directory
            try:
                script_path.resolve().relative_to(Path(self.research_dir).resolve())
            except ValueError:
                raise ValueError("Script must be within research directory")

            # Use same python executable as current process
            env = os.environ.copy()
            env["PYTHONPATH"] = (
                str(self.research_dir) + os.pathsep + env.get("PYTHONPATH", "")
            )

            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=self.research_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1,  # Line buffered
                universal_newlines=True,
            )
            self.active_processes[name] = proc

            # Collect output in a buffer instead of calling after() for every line
            def read_stream_to_buffer(stream, prefix, buffer):
                try:
                    for line in iter(stream.readline, ""):
                        if self.stop_all:
                            break
                        if line:
                            buffer.append(f"[{prefix}] {line.rstrip()}")
                except (ValueError, IOError):
                    pass
                finally:
                    stream.close()

            stdout_buffer = []
            stderr_buffer = []

            stdout_thread = threading.Thread(
                target=read_stream_to_buffer, args=(proc.stdout, name, stdout_buffer)
            )
            stderr_thread = threading.Thread(
                target=read_stream_to_buffer, args=(proc.stderr, name, stderr_buffer)
            )

            stdout_thread.start()
            stderr_thread.start()

            # Wait for process to complete
            return_code = proc.wait()

            # Ensure threads are done
            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)

            # Flush all output at once to maintain sequential order
            all_output = stdout_buffer + stderr_buffer
            if output_callback:
                output_callback(all_output)
            else:
                for line in all_output:
                    self._log(line)

            # Parse experiment results from output for visualization
            self._parse_experiment_results(name, all_output)

            status = "Success" if return_code == 0 else f"Failed (Code {return_code})"
            color = "#2ecc71" if return_code == 0 else "#e74c3c"

            self.after(0, lambda s=status, c=color: self._finish_experiment(name, s, c))

        except Exception as e:
            self.after(0, lambda ex=str(e): self._log(f"ERROR: {ex}", "#e74c3c"))
            self.after(0, lambda: self._finish_experiment(name, "Error", "#e74c3c"))
        finally:
            if name in self.active_processes:
                del self.active_processes[name]

    def _parse_experiment_results(self, experiment_name: str, output_lines: List[str]):
        """Parse experiment results from stdout output for visualization with validation."""
        results = {}
        validation_errors = []

        try:
            # Parse metrics from output lines
            for raw_line in output_lines:
                # Remove the prefix like "[Visual Search] " if present
                line = raw_line.strip()
                if line.startswith(f"[{experiment_name}] "):
                    line = line[len(f"[{experiment_name}] ") :]
                elif "] " in line and line.startswith("["):
                    # Handle other experiment name formats
                    line = line.split("] ", 1)[1] if "] " in line else line

                # Parse standard APGI metrics (with percentages)
                if "Ignition Rate:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("%")
                        value = float(value_str)
                        # Validate value is in reasonable range
                        if not (0 <= value <= 100):
                            validation_errors.append(
                                f"Ignition Rate out of range: {value}"
                            )
                        results["ignition_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Ignition Rate: {line} - {e}"
                        )

                elif "Mean Surprise:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if not (0 <= value <= 1000):
                            validation_errors.append(
                                f"Mean Surprise out of range: {value}"
                            )
                        results["mean_surprise"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Mean Surprise: {line} - {e}"
                        )

                elif "Metabolic Cost:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if value < 0:
                            validation_errors.append(
                                f"Metabolic Cost negative: {value}"
                            )
                        results["metabolic_cost"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Metabolic Cost: {line} - {e}"
                        )

                elif "Mean Somatic Marker:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if not (-10 <= value <= 10):
                            validation_errors.append(
                                f"Mean Somatic Marker out of range: {value}"
                            )
                        results["mean_somatic_marker"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Mean Somatic Marker: {line} - {e}"
                        )

                elif "Mean Threshold:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if value <= 0:
                            validation_errors.append(
                                f"Mean Threshold non-positive: {value}"
                            )
                        results["mean_threshold"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Mean Threshold: {line} - {e}"
                        )

                # Parse experiment-specific primary metrics
                elif "conjunction_present_slope:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["primary_metric"] = value
                        results["conjunction_present_slope"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse conjunction_present_slope: {line} - {e}"
                        )

                elif "overall_accuracy:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("%")
                        value = float(value_str)
                        results["primary_metric"] = value
                        results["overall_accuracy"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse overall_accuracy: {line} - {e}"
                        )

                elif "benchmark_accuracy:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["primary_metric"] = value
                        results["benchmark_accuracy"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse benchmark_accuracy: {line} - {e}"
                        )

                elif "accuracy:" in line.lower():  # Primary accuracy metric
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("%")
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["accuracy"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse accuracy: {line} - {e}")

                elif "d_prime:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["primary_metric"] = value
                        results["d_prime"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse d_prime: {line} - {e}")

                elif "net_score:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["primary_metric"] = value
                        results["net_score"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse net_score: {line} - {e}")

                elif "masking_effect_ms:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["primary_metric"] = value
                        results["masking_effect_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse masking_effect_ms: {line} - {e}"
                        )

                elif "alternation_rate:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["primary_metric"] = value
                        results["alternation_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse alternation_rate: {line} - {e}"
                        )

                # Parse APGI-prefixed metrics (from experiments that use apgi_ prefix)
                elif "apgi_ignition_rate:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("%")
                        value = float(value_str)
                        results["ignition_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse apgi_ignition_rate: {line} - {e}"
                        )

                elif "apgi_mean_surprise:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["mean_surprise"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse apgi_mean_surprise: {line} - {e}"
                        )

                elif "apgi_metabolic_cost:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["metabolic_cost"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse apgi_metabolic_cost: {line} - {e}"
                        )

                elif "apgi_mean_somatic_marker:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["mean_somatic_marker"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse apgi_mean_somatic_marker: {line} - {e}"
                        )

                elif "apgi_mean_threshold:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["mean_threshold"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse apgi_mean_threshold: {line} - {e}"
                        )

                # Parse basic experiment metrics for Change Blindness and similar
                elif "Detection Rate:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip().rstrip("%")
                        value = float(value_str) / 100  # Convert percentage to decimal
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["detection_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Detection Rate: {line} - {e}"
                        )

                elif "detection_rate:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["detection_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse detection_rate: {line} - {e}"
                        )

                elif "gating_threshold:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["gating_threshold"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse gating_threshold: {line} - {e}"
                        )

                elif "metabolic_cost_ratio:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["metabolic_cost_ratio"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse metabolic_cost_ratio: {line} - {e}"
                        )

                elif "multisensory_gain_ms:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["multisensory_gain_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse multisensory_gain_ms: {line} - {e}"
                        )

                elif "global_advantage_ms:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["global_advantage_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse global_advantage_ms: {line} - {e}"
                        )

                elif "interference_effect_ms:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["interference_effect_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse interference_effect_ms: {line} - {e}"
                        )

                elif "flanker_effect_ms:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["flanker_effect_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse flanker_effect_ms: {line} - {e}"
                        )

                elif "ssrt_ms:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["ssrt_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse ssrt_ms: {line} - {e}")

                elif "search_slope_ms_per_item:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["search_slope_ms_per_item"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse search_slope_ms_per_item: {line} - {e}"
                        )

                elif "simon_effect_ms:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["simon_effect_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse simon_effect_ms: {line} - {e}"
                        )

                elif "priming_effect_ms:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["priming_effect_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse priming_effect_ms: {line} - {e}"
                        )

                elif "learning_effect_ms:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["learning_effect_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse learning_effect_ms: {line} - {e}"
                        )

                elif "validity_effect_ms:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["validity_effect_ms"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse validity_effect_ms: {line} - {e}"
                        )

                elif "mean_error_percent:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["mean_error_percent"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse mean_error_percent: {line} - {e}"
                        )

                elif "path_efficiency:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["path_efficiency"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse path_efficiency: {line} - {e}"
                        )

                elif "learning_rate:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        if "primary_metric" not in results:
                            results["primary_metric"] = value
                        results["learning_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse learning_rate: {line} - {e}"
                        )

                # Additional visual search specific metrics
                elif "feature_present_slope:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["feature_present_slope"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse feature_present_slope: {line} - {e}"
                        )

                elif "slope_ratio:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["slope_ratio"] = value
                    except (ValueError, IndexError) as e:
                        self._log(f"[DEBUG] Failed to parse slope_ratio: {line} - {e}")

            # Store results if any metrics were found
            if results:
                self.experiment_results[experiment_name] = results
                self._log(
                    f"[VISUALIZATION] Parsed {len(results)} metrics for {experiment_name}: {list(results.keys())}"
                )
                # Log validation errors if any
                if validation_errors:
                    self._log(
                        f"[WARNING] Validation issues found for {experiment_name}:"
                    )
                    for error in validation_errors:
                        self._log(f"  - {error}")
            else:
                self._log(f"[VISUALIZATION] No metrics found for {experiment_name}")
                self._log(f"[DEBUG] Checked {len(output_lines)} output lines")

        except Exception as e:
            self._log(f"[ERROR] Failed to parse results: {e}")

    def _finish_experiment(self, name, status, color):
        self.running_experiments.discard(name)
        if name in self.experiment_buttons:
            self.experiment_buttons[name].configure(state="normal")
            self.status_indicators[name].configure(text=status, text_color=color)
        self._log(f"[FINISHED] {name}: {status}")

    def _run_all(self):
        if self.running_experiments:
            if not messagebox.askyesno(
                "Confirm", "Some experiments are already running. Run all sequentially?"
            ):
                return

        self.stop_all = False
        self._log("\n" + "#" * 50)
        self._log("### STARTING SEQUENTIAL RUN OF ALL EXPERIMENTS ###")
        self._log("#" * 50 + "\n")

        thread = threading.Thread(target=self._run_all_sequential)
        thread.daemon = True
        thread.start()

    def _run_all_sequential(self):
        """Run all experiments sequentially with proper synchronization."""
        for i, (name, script) in enumerate(self.experiments):
            if self.stop_all:
                break

            # Create a proper closure to avoid race condition
            experiment_data = (name, script)

            # Use main thread to start run with proper closure
            self.after(
                0, lambda data=experiment_data: self._run_experiment(data[0], data[1])
            )

            # Wait for this experiment to finish with timeout
            wait_time = 0
            max_wait_time = 300  # 5 minutes max per experiment
            while (
                name in self.running_experiments
                and not self.stop_all
                and wait_time < max_wait_time
            ):
                time.sleep(0.5)
                wait_time += 0.5

            # Check if experiment timed out
            if wait_time >= max_wait_time and name in self.running_experiments:
                self.after(
                    0,
                    lambda n=name: self._log(
                        f"\n!!! EXPERIMENT {n} TIMED OUT !!!", "#e74c3c"
                    ),
                )
                # Force stop the timed out experiment
                if name in self.active_processes:
                    self._stop_experiment(name)

            if self.stop_all:
                self.after(
                    0, lambda: self._log("\n!!! SEQUENTIAL RUN ABORTED !!!", "#e74c3c")
                )
                break

        self.after(
            0, lambda: self._log("\n### ALL EXPERIMENTS COMPLETE ###", "#2ecc71")
        )

    def _stop_all(self):
        self.stop_all = True
        for name, proc in list(self.active_processes.items()):
            try:
                proc.terminate()
            except Exception:
                pass
        self._log("\n[STOPPING] Termination signal sent to all active processes...")

    def _check_dependencies(self):
        """Check if all required dependencies are installed."""
        self._log("\n" + "=" * 50)
        self._log("CHECKING DEPENDENCIES")
        self._log("=" * 50)

        missing = []
        installed = []

        for module, description in {
            **CORE_DEPENDENCIES,
            **OPTIONAL_DEPENDENCIES,
        }.items():
            try:
                if module == "PIL":
                    spec = importlib.util.find_spec("PIL")
                elif module == "sklearn":
                    spec = importlib.util.find_spec("sklearn")
                else:
                    spec = importlib.util.find_spec(module)

                if spec is None:
                    missing.append((module, description))
                else:
                    installed.append((module, description))
                    self._log(f"✅ {description}")
            except Exception as e:
                missing.append((module, description))
                self._log(f"❌ {description} - Error: {e}")

        if missing:
            self._log(f"\n⚠️ Missing {len(missing)} dependencies:")
            for module, desc in missing:
                self._log(f"   - {desc}")
        else:
            self._log("\n✅ All dependencies are installed!")

        self._log("=" * 50)

    def _repair_dependencies(self):
        """Install missing dependencies using pip."""
        self._check_dependencies()

        # Check again for missing deps
        missing = []
        for module, description in {
            **CORE_DEPENDENCIES,
            **OPTIONAL_DEPENDENCIES,
        }.items():
            try:
                if module == "PIL":
                    spec = importlib.util.find_spec("PIL")
                elif module == "sklearn":
                    spec = importlib.util.find_spec("sklearn")
                else:
                    spec = importlib.util.find_spec(module)

                if spec is None:
                    missing.append(module)
            except Exception:
                missing.append(module)

        if not missing:
            messagebox.showinfo(
                "Dependencies", "All dependencies are already installed!"
            )
            return

        if not messagebox.askyesno(
            "Install Dependencies",
            f"Install {len(missing)} missing packages?\n\nThis will run:\npip install {' '.join(missing)}",
        ):
            return

        self._log(f"\n🔧 Installing {len(missing)} packages...")

        # Safe package mapping for known packages
        package_map = {
            "numpy": "numpy",
            "pandas": "pandas",
            "matplotlib": "matplotlib",
            "customtkinter": "customtkinter",
            "scipy": "scipy",
            "torch": "torch",
            "sklearn": "scikit-learn",
            "PIL": "Pillow",
        }

        # Validate and sanitize package names
        packages = []
        for m in missing:
            if m in package_map:
                packages.append(package_map[m])
            elif re.match(
                r"^[a-zA-Z][a-zA-Z0-9_-]*$", m
            ):  # Basic package name validation
                packages.append(m)
            else:
                logger.warning(f"Skipping invalid package name: {m}")

        if not packages:
            self.after(0, lambda: self._log("No valid packages to install", "#e74c3c"))
            return

        def install():
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "pip", "install"] + packages,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                stdout, stderr = proc.communicate()

                if proc.returncode == 0:
                    self.after(
                        0,
                        lambda: self._log(
                            "✅ Installation complete! Please restart the GUI."
                        ),
                    )
                else:
                    self.after(
                        0,
                        lambda: self._log(
                            f"❌ Installation failed:\n{stderr}", "#e74c3c"
                        ),
                    )
            except Exception as install_error:
                self.after(
                    0,
                    lambda err=str(install_error): self._log(
                        f"❌ Error: {err}", "#e74c3c"
                    ),
                )

        thread = threading.Thread(target=install)
        thread.daemon = True
        thread.start()

    def change_appearance_mode(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def _create_visualization_panel(self, parent_frame):
        """Create an embedded matplotlib visualization panel."""
        # Create figure and canvas
        self.current_figure = Figure(figsize=(8, 4), dpi=100, facecolor="#2b2b2b")
        self.current_canvas = FigureCanvasTkAgg(
            self.current_figure, master=parent_frame
        )
        self.current_canvas.draw()
        self.current_canvas.get_tk_widget().pack(
            fill="both", expand=True, padx=5, pady=5
        )

        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.current_canvas, parent_frame)
        toolbar.update()

    def _plot_experiment_results(self, experiment_name: str, results: dict):
        """Plot experiment results in the embedded visualization panel."""
        if self.current_figure is None:
            return

        # Clear previous plots
        self.current_figure.clear()

        # Create subplots based on available data
        ax1 = self.current_figure.add_subplot(121)
        ax2 = self.current_figure.add_subplot(122)

        # Set dark theme colors
        for ax in [ax1, ax2]:
            ax.set_facecolor("#2b2b2b")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_color("white")

        # Plot 1: Primary Metric (if available)
        if "primary_metric" in results:
            metric_value = results["primary_metric"]
            ax1.bar(["Primary Metric"], [metric_value], color="#3498db", alpha=0.7)
            ax1.set_ylim(0, max(1.0, metric_value * 1.2))
            ax1.set_title(f"{experiment_name} - Primary Metric")
            ax1.set_ylabel("Score")

        # Plot 2: APGI Metrics (if available)
        apgi_metrics = {}
        for key in [
            "ignition_rate",
            "mean_surprise",
            "metabolic_cost",
            "mean_threshold",
        ]:
            if key in results:
                apgi_metrics[key.replace("_", " ").title()] = results[key]

        if apgi_metrics:
            ax2.bar(
                range(len(apgi_metrics)),
                list(apgi_metrics.values()),
                color=["#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"],
                alpha=0.7,
            )
            ax2.set_xticks(range(len(apgi_metrics)))
            ax2.set_xticklabels(list(apgi_metrics.keys()), rotation=45, ha="right")
            ax2.set_title("APGI Dynamics Metrics")
            ax2.set_ylabel("Value")

        self.current_figure.tight_layout()
        if self.current_canvas is not None:
            self.current_canvas.draw()

    def _show_results_visualization(self, experiment_name: str):
        """Open a visualization window for experiment results."""
        viz_window = ctk.CTkToplevel(self)
        viz_window.title(f"Results Visualization - {experiment_name}")
        viz_window.geometry("900x500")
        viz_window.transient(self)

        # Create visualization panel in the new window
        viz_frame = ctk.CTkFrame(viz_window)
        viz_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create window-local figure/canvas (not shared with main panel)
        viz_figure = Figure(figsize=(10, 5), dpi=100, facecolor="#2b2b2b")
        viz_canvas = FigureCanvasTkAgg(viz_figure, master=viz_frame)
        viz_canvas.draw()
        viz_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Add toolbar
        toolbar = NavigationToolbar2Tk(viz_canvas, viz_frame)
        toolbar.update()

        # Plot results if available (use window-local figure)
        if experiment_name in self.experiment_results:
            results = self.experiment_results[experiment_name]
            self._log(
                f"[VIZ] Showing visualization for {experiment_name} with {len(results)} metrics"
            )
            # Plot directly on viz_figure instead of using shared self.current_figure
            viz_figure.clear()
            ax1 = viz_figure.add_subplot(121)
            ax2 = viz_figure.add_subplot(122)

            # Set dark theme colors
            for ax in [ax1, ax2]:
                ax.set_facecolor("#2b2b2b")
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.title.set_color("white")
                for spine in ax.spines.values():
                    spine.set_color("white")

            # Plot 1: Primary Metric
            if "primary_metric" in results:
                metric_value = results["primary_metric"]
                ax1.bar(["Primary Metric"], [metric_value], color="#3498db", alpha=0.7)
                ax1.set_ylim(0, max(1.0, metric_value * 1.2))
                ax1.set_title(f"{experiment_name} - Primary Metric")
                ax1.set_ylabel("Score")

            # Plot 2: APGI Metrics
            apgi_metrics = {}
            for key in [
                "ignition_rate",
                "mean_surprise",
                "metabolic_cost",
                "mean_threshold",
            ]:
                if key in results:
                    apgi_metrics[key.replace("_", " ").title()] = results[key]

            if apgi_metrics:
                ax2.bar(
                    range(len(apgi_metrics)),
                    list(apgi_metrics.values()),
                    color=["#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"],
                    alpha=0.7,
                )
                ax2.set_xticks(range(len(apgi_metrics)))
                ax2.set_xticklabels(list(apgi_metrics.keys()), rotation=45, ha="right")
                ax2.set_title("APGI Dynamics Metrics")
                ax2.set_ylabel("Value")

            viz_figure.tight_layout()
            viz_canvas.draw()
        else:
            # Show placeholder with debug info
            ax = viz_figure.add_subplot(111)
            ax.set_facecolor("#2b2b2b")

            # Check if we have any results at all
            available_experiments = list(self.experiment_results.keys())
            debug_info = (
                f"\n\nAvailable experiments with results: {available_experiments}"
                if available_experiments
                else "\n\nNo experiment results stored yet."
            )

            message = f"No results available for '{experiment_name}'.\nRun the experiment first to see visualization.{debug_info}"
            ax.text(
                0.5,
                0.5,
                message,
                ha="center",
                va="center",
                fontsize=12,
                color="white",
                transform=ax.transAxes,
                wrap=True,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            viz_canvas.draw()
            self._log(
                f"[VIZ] No results available for {experiment_name}. Available: {available_experiments}"
            )

        # No need to restore - we used window-local variables
        def on_close():
            viz_window.destroy()

        viz_window.protocol("WM_DELETE_WINDOW", on_close)

    def _show_file_menu(self) -> None:
        """Show File menu with options."""
        file_menu = ctk.CTkToplevel(self)
        file_menu.title("File")
        file_menu.geometry("200x150")
        file_menu.transient(self)
        ctk.CTkButton(file_menu, text="Exit", command=self.quit).pack(pady=10)

    def _show_edit_menu(self) -> None:
        """Show Edit menu with options."""
        edit_menu = ctk.CTkToplevel(self)
        edit_menu.title("Edit")
        edit_menu.geometry("200x150")
        edit_menu.transient(self)
        ctk.CTkLabel(edit_menu, text="Edit options - Coming soon").pack(pady=20)

    def _show_view_menu(self) -> None:
        """Show View menu with options."""
        view_menu = ctk.CTkToplevel(self)
        view_menu.title("View")
        view_menu.geometry("200x150")
        view_menu.transient(self)
        ctk.CTkButton(
            view_menu,
            text="Toggle Appearance",
            command=lambda: self.change_appearance_mode(
                "Light" if ctk.get_appearance_mode() == "Dark" else "Dark"
            ),
        ).pack(pady=10)

    def _show_help_menu(self) -> None:
        """Show Help menu with options."""
        help_menu = ctk.CTkToplevel(self)
        help_menu.title("Help")
        help_menu.geometry("300x200")
        help_menu.transient(self)
        ctk.CTkLabel(
            help_menu,
            text="APGI Experiment Runner",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=10)
        ctk.CTkLabel(help_menu, text="Version 2.0 - APGI Research Hub").pack(pady=5)
        ctk.CTkButton(help_menu, text="Close", command=help_menu.destroy).pack(pady=20)


if __name__ == "__main__":
    app = ExperimentRunnerGUI()
    app.mainloop()
