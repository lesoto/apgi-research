"""APGI Experiment Runner GUI (Premium Edition)
Modernized for apgi-research directory with CustomTkinter.
"""

from tkinter import messagebox
import customtkinter as ctk  # type: ignore
import subprocess
import threading
import os
import sys
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

        # Application state
        self.running_experiments: Set[str] = set()
        self.experiment_cards: Dict[str, ctk.CTkFrame] = {}
        self.experiment_buttons: Dict[str, ctk.CTkButton] = {}
        self.status_indicators: Dict[str, ctk.CTkLabel] = {}
        self.active_processes: Dict[str, subprocess.Popen] = {}
        self.stop_all = False
        self.current_figure = None
        self.current_canvas = None
        self.experiment_results: Dict[str, dict] = {}

        # Find experiments
        self.experiments = self._find_experiments()

        self._setup_ui()

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

    def _setup_ui(self) -> None:
        # Configure grid layout (1x2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create navigation frame (sidebar)
        self.navigation_frame = ctk.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
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

        # Dependency management section
        self.deps_frame = ctk.CTkFrame(self.navigation_frame, fg_color="transparent")
        self.deps_frame.grid(row=3, column=0, padx=20, pady=(20, 10), sticky="ew")

        self.repair_deps_button = ctk.CTkButton(
            self.deps_frame,
            text="🔧 Repair Installation",
            command=self._repair_dependencies,
            height=40,
            font=ctk.CTkFont(size=12),
            fg_color="#e67e22",
            hover_color="#d35400",
        )
        self.repair_deps_button.pack_forget()  # Hidden button but functionality preserved

        self.deps_status_label = ctk.CTkLabel(
            self.deps_frame,
            text="Status: Unknown",
            font=ctk.CTkFont(size=10),
            text_color="gray",
        )
        self.deps_status_label.pack(pady=(5, 0))

        # Auto-check dependencies on startup
        self.after(100, self._check_dependencies)

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
        self.appearance_mode_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(
            self.navigation_frame,
            values=["Dark", "Light", "System"],
            command=self.change_appearance_mode,
        )
        self.appearance_mode_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        # Create main scrollable area for experiments
        self.scrollable_frame = ctk.CTkScrollableFrame(
            self, label_text=f"Research Experiments ({len(self.experiments)})"
        )
        self.scrollable_frame.grid(
            row=0, column=1, padx=(20, 10), pady=(20, 10), sticky="nsew"
        )
        self.scrollable_frame.grid_columnconfigure((0, 1), weight=1)

        # Populate experiments
        for i, (name, script) in enumerate(self.experiments):
            self._create_experiment_card(self.scrollable_frame, name, script, i)

        # Create output console (bottom)
        self.console_frame = ctk.CTkFrame(self, height=250)
        self.console_frame.grid(
            row=1, column=0, columnspan=2, padx=20, pady=(10, 20), sticky="nsew"
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
        """Execute script with optional output callback."""
        try:
            # Use same python executable as current process
            env = os.environ.copy()
            env["PYTHONPATH"] = (
                str(self.research_dir) + os.pathsep + env.get("PYTHONPATH", "")
            )

            proc = subprocess.Popen(
                [sys.executable, script],
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
        """Parse experiment results from stdout output for visualization."""
        results = {}

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
                        results["ignition_rate"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Ignition Rate: {line} - {e}"
                        )

                elif "Mean Surprise:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["mean_surprise"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Mean Surprise: {line} - {e}"
                        )

                elif "Metabolic Cost:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        if value_str:  # Skip empty values
                            value = float(value_str)
                            results["metabolic_cost"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Metabolic Cost: {line} - {e}"
                        )

                elif "Mean Somatic Marker:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
                        results["mean_somatic_marker"] = value
                    except (ValueError, IndexError) as e:
                        self._log(
                            f"[DEBUG] Failed to parse Mean Somatic Marker: {line} - {e}"
                        )

                elif "Mean Threshold:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        value = float(value_str)
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

                elif (
                    "accuracy:" in line and "accuracy:" not in line.lower()
                ):  # Primary accuracy metric
                    try:
                        value_str = line.split(":", 1)[1].strip()
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
        for name, script in self.experiments:
            if self.stop_all:
                break

            # Use main thread to start run
            self.after(0, lambda n=name, s=script: self._run_experiment(n, s))

            # Wait for this one to finish
            while name in self.running_experiments and not self.stop_all:
                time.sleep(0.5)

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
            self.deps_status_label.configure(
                text=f"Status: {len(missing)} missing", text_color="#e74c3c"
            )
            self._log(f"\n⚠️ Missing {len(missing)} dependencies:")
            for module, desc in missing:
                self._log(f"   - {desc}")
        else:
            self.deps_status_label.configure(
                text="Status: All installed", text_color="#2ecc71"
            )
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

        def install():
            try:
                # Map module names to package names
                package_map = {
                    "PIL": "Pillow",
                    "sklearn": "scikit-learn",
                }
                packages = [package_map.get(m, m) for m in missing]

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
                    self.after(
                        0,
                        lambda: self.deps_status_label.configure(
                            text="Status: Installed (restart needed)",
                            text_color="#f39c12",
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

        # Store previous figure/canvas state
        prev_figure = self.current_figure
        prev_canvas = self.current_canvas

        # Create new figure for this window
        self.current_figure = Figure(figsize=(10, 5), dpi=100, facecolor="#2b2b2b")
        self.current_canvas = FigureCanvasTkAgg(self.current_figure, master=viz_frame)
        self.current_canvas.draw()
        self.current_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.current_canvas, viz_frame)
        toolbar.update()

        # Plot results if available
        if experiment_name in self.experiment_results:
            results = self.experiment_results[experiment_name]
            self._log(
                f"[VIZ] Showing visualization for {experiment_name} with {len(results)} metrics"
            )
            self._plot_experiment_results(experiment_name, results)
        else:
            # Show placeholder with debug info
            ax = self.current_figure.add_subplot(111)
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
            self.current_canvas.draw()
            self._log(
                f"[VIZ] No results available for {experiment_name}. Available: {available_experiments}"
            )

        # Restore original figure/canvas when window closes
        def on_close():
            self.current_figure = prev_figure
            self.current_canvas = prev_canvas
            viz_window.destroy()

        viz_window.protocol("WM_DELETE_WINDOW", on_close)


if __name__ == "__main__":
    app = ExperimentRunnerGUI()
    app.mainloop()
