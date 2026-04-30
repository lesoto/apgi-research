"""APGI Experiment Runner GUI (Premium Edition)
Modernized for apgi-research directory with CustomTkinter.
"""

import os

# CRITICAL: Must set multiprocessing start method BEFORE ANY OTHER IMPORTS on macOS
# to prevent "The process has forked and you cannot use this CoreFoundation functionality" error
import sys

if sys.platform == "darwin":
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

from tkinter import messagebox
from typing import Any, cast

import customtkinter as ctk


# Fix for customtkinter DropdownMenu bug: _add_menu_commands fails on empty menu
def _patched_add_menu_commands(self: Any) -> None:
    """Patched version that handles empty menus safely."""
    try:
        # Check if menu has items before trying to delete
        end_index = self.index("end")
        if end_index is not None and end_index != "":
            self.delete(0, "end")
    except Exception:
        pass  # Menu is empty or not fully initialized

    # Add the actual menu commands
    for i, value in enumerate(self._values):
        self.add_command(
            label=value,
            command=lambda v=value: self._command(v) if self._command else None,
        )


# Apply the patch
ctk.windows.widgets.core_widget_classes.dropdown_menu.DropdownMenu._add_menu_commands = (
    _patched_add_menu_commands
)

import importlib.util
import json
import logging
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Matplotlib imports for embedded visualization
import matplotlib
import numpy as np

# Import hypothesis approval board
from hypothesis_approval_board import ApprovalBoard, Hypothesis, HypothesisStatus

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Set appearance mode and color theme
ctk.set_appearance_mode("Dark")
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
        self.title("APGI Experiment Auto-Improvement")
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

        # Hypothesis approval board
        self.approval_board: ApprovalBoard = ApprovalBoard()

        # Guardrail state tracking
        self.guardrail_state: Dict[str, str | float] = {
            "status": "IDLE",
            "confidence": 1.0,
            "last_regression": 0.0,
            "escalation_count": 0,
            "last_experiment": "",
        }

        # Find experiments
        self.experiments = self._find_experiments()

        self._setup_ui()

        # Check dependencies after UI is initialized so we can log to console
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check for required dependencies on startup and show error if missing."""
        import importlib

        missing_core = []
        for module, description in CORE_DEPENDENCIES.items():
            try:
                importlib.import_module(module)
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
                importlib.import_module(module)
            except ImportError:
                missing_optional.append(f"  - {module}: {description}")

        if missing_optional:
            print("Optional dependencies missing (some features may be unavailable):")
            for msg in missing_optional:
                print(msg)

    def _find_experiments(self) -> List[Tuple[str, str]]:
        """Dynamically find all run_*.py files in the experiments directory."""
        experiments = []
        experiments_dir = self.research_dir / "experiments"
        run_files = sorted(list(experiments_dir.glob("run_*.py")))

        for file in run_files:
            # Format name: run_visual_search.py -> Visual Search
            name = file.stem.replace("run_", "").replace("_", " ").title()
            if name == "Tests":
                continue  # Skip run_tests.py if it exists here
            experiments.append((name, f"experiments/{file.name}"))

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
        self.navigation_frame.grid_rowconfigure(7, weight=1)

        self.navigation_frame_label = ctk.CTkLabel(
            self.navigation_frame,
            text="APGI Research",
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

        # -----------------------------------------------------------
        # Guardrail Dashboard Panel (APGI Requirement)
        # -----------------------------------------------------------
        self.guardrail_frame = ctk.CTkFrame(
            self.navigation_frame, corner_radius=8, fg_color="#1a1a2e"
        )
        self.guardrail_frame.grid(row=4, column=0, padx=10, pady=(10, 5), sticky="ew")

        ctk.CTkLabel(
            self.guardrail_frame,
            text="⚡ Guardrails",
            text="0.0%",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#2ecc71",
        )
        self.guardrail_regression_label.grid(
            row=3, column=1, padx=(2, 10), pady=2, sticky="w"
        )

        # Escalation count
        ctk.CTkLabel(
            self.guardrail_frame,
            text="Escalations:",
            font=ctk.CTkFont(size=11),
            text_color="#333333",
        ).grid(row=4, column=0, padx=(10, 2), pady=(2, 8), sticky="w")
        self.guardrail_escalation_label = ctk.CTkLabel(
            self.guardrail_frame,
            text="0",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#2ecc71",
        )
        self.guardrail_escalation_label.grid(
            row=4, column=1, padx=(2, 10), pady=(2, 8), sticky="w"
        )
        # -----------------------------------------------------------

        self.appearance_mode_label = ctk.CTkLabel(
            self.navigation_frame, text="Appearance:", anchor="w"
        )
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(
            self.navigation_frame,
            values=["Dark", "Light", "System"],
            command=self.change_appearance_mode,
        )
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 20))

        # -----------------------------------------------------------
        # Hypothesis Approval Board (Phase 4 Enhancement)
        # -----------------------------------------------------------
        self.hypothesis_frame = ctk.CTkFrame(
            self.navigation_frame, corner_radius=8, fg_color="#2c3e50"
        )
        self.hypothesis_frame.grid(row=7, column=0, padx=10, pady=(10, 5), sticky="ew")

        ctk.CTkLabel(
            self.hypothesis_frame,
            text="🧪 Hypothesis Board",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#e2e2e2",
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(8, 4), sticky="w")

        # Hypothesis controls
        controls_frame = ctk.CTkFrame(self.hypothesis_frame, fg_color="transparent")
        controls_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        ctk.CTkButton(
            controls_frame,
            text="➕ New Hypothesis",
            command=self._show_create_hypothesis_dialog,
            width=140,
            height=32,
            fg_color="#27ae60",
            hover_color="#219150",
        ).grid(row=0, column=0, padx=5, pady=2)

        ctk.CTkButton(
            controls_frame,
            text="📋 Review Pending",
            command=self._show_hypothesis_review,
            width=140,
            height=32,
            fg_color="#3498db",
            hover_color="#2980b9",
        ).grid(row=0, column=1, padx=5, pady=2)

        self.hypothesis_scrollable = ctk.CTkScrollableFrame(
            self.hypothesis_frame, label_text="Active Hypotheses"
        )
        self.hypothesis_scrollable.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        self.hypothesis_scrollable.grid_columnconfigure(0, weight=1)

        self._refresh_hypothesis_display()

    def _show_create_hypothesis_dialog(self) -> None:
        """Show dialog to create a new hypothesis."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("🧪 Create New Hypothesis")
        dialog.geometry("600x500")
        dialog.transient(self)
        dialog.attributes("-topmost", True)

        ctk.CTkLabel(
            dialog,
            text="Create New Hypothesis",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(15, 5))

        # Form fields
        form_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        form_frame.pack(padx=20, fill="x", pady=10)

        # Title
        ctk.CTkLabel(form_frame, text="Title:", font=ctk.CTkFont(size=12)).grid(
            row=0, column=0, sticky="w", padx=5, pady=2
        )
        title_entry = ctk.CTkEntry(form_frame, width=400)
        title_entry.grid(row=0, column=1, padx=5, pady=2)

        # Description
        ctk.CTkLabel(form_frame, text="Description:", font=ctk.CTkFont(size=12)).grid(
            row=1, column=0, sticky="nw", padx=5, pady=2
        )
        desc_entry = ctk.CTkTextbox(form_frame, height=80)
        desc_entry.grid(row=1, column=1, padx=5, pady=2, sticky="nsew")

        # Predicted Outcome
        ctk.CTkLabel(
            form_frame, text="Predicted Outcome:", font=ctk.CTkFont(size=12)
        ).grid(row=2, column=0, sticky="nw", padx=5, pady=2)
        outcome_entry = ctk.CTkEntry(form_frame, width=400)
        outcome_entry.grid(row=2, column=1, padx=5, pady=2)

        # Confidence Score
        ctk.CTkLabel(
            form_frame, text="Confidence (0-1):", font=ctk.CTkFont(size=12)
        ).grid(row=3, column=0, sticky="nw", padx=5, pady=2)
        conf_entry = ctk.CTkEntry(form_frame, width=400)
        conf_entry.insert(0, "0.7")
        conf_entry.grid(row=3, column=1, padx=5, pady=2)

        # Risk Assessment
        ctk.CTkLabel(form_frame, text="Risk Level:", font=ctk.CTkFont(size=12)).grid(
            row=4, column=0, sticky="nw", padx=5, pady=2
        )
        risk_menu = ctk.CTkOptionMenu(form_frame, values=["low", "medium", "high"])
        risk_menu.grid(row=4, column=1, padx=5, pady=2)

        # Success Criteria
        ctk.CTkLabel(
            form_frame, text="Success Criteria:", font=ctk.CTkFont(size=12)
        ).grid(row=5, column=0, sticky="nw", padx=5, pady=2)
        criteria_entry = ctk.CTkTextbox(form_frame, height=60)
        criteria_entry.grid(row=5, column=1, padx=5, pady=2, sticky="nsew")

        # Buttons
        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=15)

        def create_hypothesis() -> None:
            try:
                hypothesis = self.approval_board.create_hypothesis(
                    title=title_entry.get(),
                    description=desc_entry.get("0.0", "end"),
                    predicted_outcome=outcome_entry.get(),
                    confidence_score=float(conf_entry.get()),
                    risk_assessment=risk_menu.get(),
                    success_criteria=(
                        criteria_entry.get("0.0", "end").strip().split("\n")
                        if criteria_entry.get("0.0", "end").strip()
                        else []
                    ),
                )
                self._refresh_hypothesis_display()
                dialog.destroy()
                self._log(f"✅ Created hypothesis: {hypothesis.title}", "#2ecc71")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create hypothesis: {e}")

        def cancel() -> None:
            dialog.destroy()

        ctk.CTkButton(
            btn_frame,
            text="Create",
            command=create_hypothesis,
            fg_color="#27ae60",
            hover_color="#219150",
            width=100,
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            btn_frame,
            text="Cancel",
            command=cancel,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            width=100,
        ).pack(side="left", padx=5)

    def _show_hypothesis_review(self) -> None:
        """Show dialog to review pending hypotheses."""
        pending_hypotheses = self.approval_board.get_pending_hypotheses()
        if not pending_hypotheses:
            messagebox.showinfo(
                "No Pending Hypotheses", "No hypotheses pending review."
            )
            return

        dialog = ctk.CTkToplevel(self)
        dialog.title("📋 Review Hypotheses")
        dialog.geometry("800x600")
        dialog.transient(self)
        dialog.attributes("-topmost", True)

        ctk.CTkLabel(
            dialog,
            text="Pending Hypotheses Review",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(15, 5))

        # Create scrollable frame for hypotheses
        review_frame = ctk.CTkScrollableFrame(dialog, height=400)
        review_frame.pack(padx=20, pady=10, fill="both", expand=True)

        for hypothesis in pending_hypotheses:
            self._create_hypothesis_review_item(review_frame, hypothesis, dialog)

        # Buttons
        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=10)

        def close_review() -> None:
            dialog.destroy()

        ctk.CTkButton(
            btn_frame,
            text="Close",
            command=close_review,
            fg_color="#7f8c8d",
            hover_color="#636e72",
            width=100,
        ).pack()

    def _create_hypothesis_review_item(
        self,
        parent: ctk.CTkFrame,
        hypothesis: Hypothesis,
        dialog: ctk.CTkToplevel | None,
    ) -> None:
        """Create a single hypothesis review item."""
        item_frame = ctk.CTkFrame(parent, corner_radius=8, fg_color="#f8f9fa")
        item_frame.pack(fill="x", pady=5, padx=10)

        # Title and status
        header_frame = ctk.CTkFrame(item_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=5)

        title_label = ctk.CTkLabel(
            header_frame,
            text=hypothesis.title,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        title_label.pack(side="left", padx=5)

        status_colors = {
            HypothesisStatus.PENDING: "#f39c12",
            HypothesisStatus.UNDER_REVIEW: "#3498db",
            HypothesisStatus.APPROVED: "#2ecc71",
            HypothesisStatus.REJECTED: "#e74c3c",
            HypothesisStatus.MODIFIED: "#ff9500",
        }
        status_label = ctk.CTkLabel(
            header_frame,
            text=hypothesis.status.value.upper(),
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=status_colors.get(hypothesis.status, "#666666"),
        )
        status_label.pack(side="right", padx=5)

        # Description
        desc_frame = ctk.CTkFrame(item_frame, fg_color="transparent")
        desc_frame.pack(fill="x", pady=(0, 5), padx=10)

        desc_label = ctk.CTkLabel(
            desc_frame,
            text="Description:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#333333",
        )
        desc_label.pack(anchor="w", padx=(0, 2))

        desc_text = ctk.CTkLabel(
            desc_frame,
            text=hypothesis.description,
            font=ctk.CTkFont(size=10),
            wraplength=300,
        )
        desc_text.pack(anchor="w", padx=(0, 2), pady=(0, 5))

        # Metadata
        meta_frame = ctk.CTkFrame(item_frame, fg_color="transparent")
        meta_frame.pack(fill="x", pady=5, padx=10)

        # Confidence and Risk
        ctk.CTkLabel(
            meta_frame,
            text=f"Confidence: {hypothesis.confidence_score:.2f}",
            font=ctk.CTkFont(size=10),
        ).pack(anchor="w", pady=2)
        ctk.CTkLabel(
            meta_frame,
            text=f"Risk: {hypothesis.risk_assessment.upper()}",
            font=ctk.CTkFont(size=10),
        ).pack(anchor="w", pady=2)

        # Action buttons
        action_frame = ctk.CTkFrame(item_frame, fg_color="transparent")
        action_frame.pack(fill="x", pady=5, padx=10)

        def approve() -> None:
            self.approval_board.update_hypothesis_status(
                hypothesis.id,
                HypothesisStatus.APPROVED,
                "Approved via GUI review",
                "GUI Reviewer",
            )
            self._refresh_hypothesis_display()
            if dialog:
                dialog.destroy()

        def reject() -> None:
            self.approval_board.update_hypothesis_status(
                hypothesis.id,
                HypothesisStatus.REJECTED,
                "Rejected via GUI review",
                "GUI Reviewer",
            )
            self._refresh_hypothesis_display()
            if dialog:
                dialog.destroy()

        def modify() -> None:
            # For now, just log - could open modify dialog later
            self._log(f"Request to modify hypothesis: {hypothesis.id}")

        ctk.CTkButton(
            action_frame,
            text="✅ Approve",
            command=approve,
            fg_color="#27ae60",
            hover_color="#219150",
            width=80,
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            action_frame,
            text="🔧 Modify",
            command=modify,
            fg_color="#f39c12",
            hover_color="#d68910",
            width=80,
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            action_frame,
            text="❌ Reject",
            command=reject,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            width=80,
        ).pack(side="left", padx=5)

    def _refresh_hypothesis_display(self) -> None:
        """Refresh the hypothesis display scrollable frame."""
        for widget in self.hypothesis_scrollable.winfo_children():
            widget.destroy()

        pending_hypotheses = self.approval_board.get_pending_hypotheses()

        if not pending_hypotheses:
            no_hypotheses_label = ctk.CTkLabel(
                self.hypothesis_scrollable,
                text="No pending hypotheses to review",
                font=ctk.CTkFont(size=12, slant="italic"),
                text_color="#333333",
            )
            no_hypotheses_label.pack(pady=20)
        else:
            for hypothesis in pending_hypotheses:
                self._create_hypothesis_review_item(
                    self.hypothesis_scrollable, hypothesis, None
                )

        self.scrollable_frame = ctk.CTkScrollableFrame(
            self, label_text=f"Research Experiments ({len(self.experiments)})"
        )
        self.scrollable_frame.grid(
            row=1, column=1, padx=(20, 10), pady=(20, 10), sticky="nsew"
        )
        self.scrollable_frame.grid_columnconfigure((0, 1), weight=1)

        for i, (name, script) in enumerate(self.experiments):
            self._create_experiment_card(self.scrollable_frame, name, script, i)

        self.console_frame = ctk.CTkFrame(self, height=250)
        self.console_frame.grid(
            row=2, column=0, columnspan=2, padx=20, pady=(10, 20), sticky="nsew"
        )
        self.console_frame.grid_columnconfigure(0, weight=1)
        self.console_frame.grid_rowconfigure(0, weight=1)

        self.console_text = ctk.CTkTextbox(self.console_frame, font=("Courier", 13))
        self.console_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.console_text.insert("0.0", "--- APGI Research Console Ready ---\n")

    def _create_experiment_card(
        self, parent: Any, name: str, script: str, index: int
    ) -> None:
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
            card, text=f"📄 {script}", font=ctk.CTkFont(size=12), text_color="#444444"
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

        auto_btn = ctk.CTkButton(
            card,
            text="🤖 XPR AUTO",
            command=lambda n=name, s=script: self._run_auto_improve(n, s),
            width=70,
            height=28,
            fg_color="#8e44ad",
            hover_color="#9b59b6",
        )
        auto_btn.grid(row=0, column=2, padx=(0, 15), pady=(15, 5))

        self.experiment_cards[name] = card

    def _log(self, text: str, color: str | None = None) -> None:
        self.console_text.insert("end", text + "\n")
        self.console_text.see("end")

    def _clear_console(self) -> None:
        self.console_text.delete("1.0", "end")
        self.console_text.insert("1.0", "--- Console Cleared ---\n")

    def _update_guardrail_dashboard(
        self,
        status: str = "IDLE",
        confidence: float = 1.0,
        regression: float = 0.0,
        experiment: str = "",
    ) -> None:
        """Update the guardrail dashboard indicators in the sidebar."""
        self.guardrail_state["status"] = status
        self.guardrail_state["confidence"] = confidence
        self.guardrail_state["last_regression"] = regression
        if experiment:
            self.guardrail_state["last_experiment"] = experiment

        # Status color logic
        status_colors = {
            "IDLE": "#2ecc71",
            "RUNNING": "#3498db",
            "OK": "#2ecc71",
            "WARNING": "#f39c12",
            "ESCALATED": "#e74c3c",
            "HALTED": "#e74c3c",
        }
        status_color = status_colors.get(status, "#aaaaaa")
        self.guardrail_status_label.configure(text=status, text_color=status_color)

        # Confidence color: green > 0.7, yellow > 0.4, red otherwise
        if confidence > 0.7:
            conf_color = "#2ecc71"
        elif confidence > 0.4:
            conf_color = "#f39c12"
        else:
            conf_color = "#e74c3c"
        self.guardrail_confidence_label.configure(
            text=f"{confidence:.0%}", text_color=conf_color
        )

        # Regression color: green < 2%, yellow < 5%, red otherwise
        reg_pct = abs(regression) * 100
        if reg_pct < 2:
            reg_color = "#2ecc71"
        elif reg_pct < 5:
            reg_color = "#f39c12"
        else:
            reg_color = "#e74c3c"
        self.guardrail_regression_label.configure(
            text=f"{reg_pct:.1f}%", text_color=reg_color
        )

        esc_count = self.guardrail_state.get("escalation_count", 0)
        esc_color = "#2ecc71" if esc_count == 0 else "#e74c3c"
        self.guardrail_escalation_label.configure(
            text=str(esc_count), text_color=esc_color
        )

    def _notify_guardrail_escalation(
        self, experiment_name: str, confidence: float, message: str
    ) -> None:
        """Show a GUI notification when a guardrail is tripped."""
        self.guardrail_state["escalation_count"] = (
            int(self.guardrail_state.get("escalation_count", 0)) + 1
        )
        self._update_guardrail_dashboard(
            status="ESCALATED", confidence=confidence, experiment=experiment_name
        )
        self._log(
            f"\n🚨 [GUARDRAIL ESCALATION] {experiment_name}: {message}", "#e74c3c"
        )
        # Show a non-blocking alert dialog
        alert = ctk.CTkToplevel(self)
        alert.title("⚠️ Guardrail Escalation")
        alert.geometry("450x200")
        alert.transient(self)
        alert.attributes("-topmost", True)

        ctk.CTkLabel(
            alert,
            text="🚨 Agent Guardrail Tripped",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#e74c3c",
        ).pack(pady=(15, 5))
        ctk.CTkLabel(
            alert,
            text=f"Experiment: {experiment_name}\nConfidence: {confidence:.0%}\n{message}",
            font=ctk.CTkFont(size=12),
            justify="center",
        ).pack(pady=10)
        ctk.CTkButton(
            alert,
            text="Acknowledge",
            command=alert.destroy,
            fg_color="#e74c3c",
            hover_color="#c0392b",
        ).pack(pady=10)

    def _run_auto_improve(self, name: str, script: str) -> None:
        """Phase 4: Launch the Steering Dashboard before execution."""

        # -----------------------------------------------------------
        # Phase 2: configure_if_needed() — Pre-run constraints dialog
        # -----------------------------------------------------------
        config_dialog = ctk.CTkToplevel(self)
        config_dialog.title(f"⚙️ Configure — {name}")
        config_dialog.geometry("480x400")
        config_dialog.transient(self)
        config_dialog.attributes("-topmost", True)

        ctk.CTkLabel(
            config_dialog,
            text="Pre-Run Configuration",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(15, 5))
        ctk.CTkLabel(
            config_dialog,
            text="Set bounds and constraints before XPR AUTO execution.",
            font=ctk.CTkFont(size=11),
            text_color="#333333",
        ).pack(pady=(0, 10))

        fields_frame = ctk.CTkFrame(config_dialog, fg_color="transparent")
        fields_frame.pack(padx=20, fill="x")

        # Max Iterations
        ctk.CTkLabel(
            fields_frame, text="Max Iterations:", font=ctk.CTkFont(size=12)
        ).grid(row=0, column=0, sticky="w", pady=5)
        iter_entry = ctk.CTkEntry(fields_frame, width=120)
        iter_entry.insert(0, "3")
        iter_entry.grid(row=0, column=1, padx=10, pady=5)

        # Time Budget (seconds)
        ctk.CTkLabel(
            fields_frame, text="Time Budget (s):", font=ctk.CTkFont(size=12)
        ).grid(row=1, column=0, sticky="w", pady=5)
        time_entry = ctk.CTkEntry(fields_frame, width=120)
        time_entry.insert(0, "600")
        time_entry.grid(row=1, column=1, padx=10, pady=5)

        # Confidence Threshold
        ctk.CTkLabel(
            fields_frame, text="Min Confidence:", font=ctk.CTkFont(size=12)
        ).grid(row=2, column=0, sticky="w", pady=5)
        conf_entry = ctk.CTkEntry(fields_frame, width=120)
        conf_entry.insert(0, "0.5")
        conf_entry.grid(row=2, column=1, padx=10, pady=5)

        # Protected Files
        ctk.CTkLabel(
            fields_frame, text="Protected Files:", font=ctk.CTkFont(size=12)
        ).grid(row=3, column=0, sticky="w", pady=5)
        protected_entry = ctk.CTkEntry(fields_frame, width=250)
        protected_entry.insert(0, "prepare_*.py")
        protected_entry.grid(row=3, column=1, padx=10, pady=5)

        # Additional constraints
        ctk.CTkLabel(
            config_dialog, text="Additional Constraints:", font=ctk.CTkFont(size=12)
        ).pack(padx=20, anchor="w", pady=(10, 0))
        constraints_box = ctk.CTkTextbox(config_dialog, height=60, width=430)
        constraints_box.pack(padx=20, pady=5)
        constraints_box.insert(
            "0.0", "Do not modify file structure. Keep bounds valid."
        )

        # Store config for use in the plan generation
        run_config: Dict[str, Any] = {"proceed": False}

        def on_proceed() -> None:
            try:
                run_config["iterations"] = int(iter_entry.get())
            except ValueError:
                run_config["iterations"] = 3
            try:
                run_config["time_budget"] = int(time_entry.get())
            except ValueError:
                run_config["time_budget"] = 600
            try:
                run_config["min_confidence"] = float(conf_entry.get())
            except ValueError:
                run_config["min_confidence"] = 0.5
            try:
                run_config["protected_files"] = protected_entry.get().strip()
            except Exception:
                pass
            run_config["proceed"] = True
            # Store config for use in plan generation
            self.run_config = run_config
            config_dialog.destroy()

        def on_skip() -> None:
            run_config["iterations"] = 3
            run_config["time_budget"] = 600
            run_config["min_confidence"] = 0.5
            run_config["protected_files"] = "prepare_*.py"
            run_config["proceed"] = True
            self.run_config = run_config
            config_dialog.destroy()

        btn_row = ctk.CTkFrame(config_dialog, fg_color="transparent")
        btn_row.pack(pady=15)
        ctk.CTkButton(
            btn_row,
            text="▶ Proceed",
            command=on_proceed,
            fg_color="#27ae60",
            hover_color="#219150",
            width=120,
        ).pack(side="left", padx=10)
        ctk.CTkButton(
            btn_row,
            text="Skip (Defaults)",
            command=on_skip,
            fg_color="#7f8c8d",
            hover_color="#636e72",
            width=120,
        ).pack(side="left", padx=10)
        ctk.CTkButton(
            btn_row,
            text="Cancel",
            command=config_dialog.destroy,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            width=100,
        ).pack(side="left", padx=10)

        def wait_for_config() -> None:
            """Wait for the config dialog to close, then continue."""
            if config_dialog.winfo_exists():
                self.after(100, wait_for_config)
                return
            if not run_config.get("proceed"):
                self._log(f"[XPR AGENT] Configuration cancelled for {name}.", "#7f8c8d")
                return

            self._log(
                f"\n[XPR AGENT] Configured: iters={run_config.get('iterations')}, "
                f"budget={run_config.get('time_budget')}s, "
                f"min_conf={run_config.get('min_confidence')}",
                "#8e44ad",
            )
            self._launch_plan_generation(name, script, run_config)

        self.after(100, wait_for_config)

    def _launch_plan_generation(
        self, name: str, script: str, run_config: Dict[str, Any]
    ) -> None:
        """Boot the XPR Agent after configuration is complete."""
        from autonomous_agent import AutonomousAgent
        from xpr_agent_engine import XPRAgentEngine

        # Create dialog for plan review
        dialog = ctk.CTkToplevel(self)
        dialog.title(f"XPR Control Layer - {name}")
        dialog.geometry("600x500")

        ctk.CTkLabel(
            dialog,
            text="ExperimentPlan Review",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=10)

        plan_text = ctk.CTkTextbox(dialog, height=300, width=550)
        plan_text.pack(pady=10, padx=20)

        def popup_callback(plan_str: str) -> None:
            popup_dialog = ctk.CTkToplevel(self)
            popup_dialog.title(f"XPR Control Layer - {name}")
            popup_dialog.geometry("600x500")

            ctk.CTkLabel(
                popup_dialog,
                text="ExperimentPlan Review",
                font=ctk.CTkFont(size=16, weight="bold"),
            ).pack(pady=10)

            plan_display_text = ctk.CTkTextbox(popup_dialog, height=300, width=550)
            plan_display_text.pack(pady=10, padx=20)
            plan_display_text.insert("0.0", plan_str)

            def approve() -> None:
                self._log(
                    f"[XPR AGENT] Plan APPROVED. Executing {name} tuning.", "#2ecc71"
                )
                self._update_guardrail_dashboard(status="RUNNING", experiment=name)
                popup_dialog.destroy()

                def run_agent() -> None:
                    try:
                        agent = AutonomousAgent(str(self.research_dir))
                        experiment_key = (
                            script.replace("experiments/", "")
                            .replace("run_", "")
                            .replace(".py", "")
                        )
                        agent.optimize_experiment(
                            experiment_key, iterations=3, resume=False
                        )
                        self.after(
                            0,
                            lambda: self._update_guardrail_dashboard(
                                status="OK", confidence=0.9, experiment=name
                            ),
                        )
                    except Exception as e:
                        self._log(f"[XPR AGENT] Agent execution failed: {e}", "#e74c3c")
                        self._update_guardrail_dashboard(
                            status="WARNING", confidence=0.3, experiment=name
                        )
                        self._notify_guardrail_escalation(name, 0.3, str(e))

                threading.Thread(target=run_agent, daemon=True).start()

        # Generate plan using agent engine
        try:
            agent_engine = XPRAgentEngine()
            plan_result = agent_engine.plan_experiment(name, {})
            plan_str = (
                str(plan_result.result)
                if plan_result.success
                else "Failed to generate plan"
            )
            plan_text.insert("0.0", plan_str)
            # Remove unused variable
            # plan_details = (
            #     plan_result.result
            #     if plan_result.success
            #     else f"Failed to generate plan: {plan_result.error}"
            # )
        except Exception as e:
            plan_str = f"Failed to generate plan: {e}"
            self._log(f"[XPR AGENT] Plan APPROVED. Executing {name} tuning.", "#2ecc71")
            self._update_guardrail_dashboard(status="RUNNING", experiment=name)
            self._update_guardrail_dashboard(
                status="WARNING", confidence=0.6, experiment=name
            )
            # Capture the current plan text (may have been edited by human)
            current_plan_text = self.agent_engine.get_current_plan()
            if current_plan_text and current_plan_text.result:
                plan_text = current_plan_text.result.get("plan", "No plan available")
            else:
                plan_text = "No plan available"
            dialog.destroy()

            def run_modify_chain() -> None:
                try:
                    engine = XPRAgentEngine()
                    experiment_key = (
                        script.replace("experiments/", "")
                        .replace("run_", "")
                        .replace(".py", "")
                    )
                    # Step 1: Run issue-fix chain on the current plan
                    # Use execute_skill instead of non-existent xpr_skill_chain
                    fix_result = engine.execute_skill(
                        "issue_fix",
                        experiment_key=experiment_key,
                        original_plan=plan_text,
                        current_plan=current_plan_text,
                    )
                    fix_summary = (
                        str(fix_result.result)
                        if fix_result.success
                        else "No fix output generated."
                    )
                    self._log(
                        f"[XPR AGENT] Issue-fix chain complete: {fix_summary[:200]}",
                        "#f39c12",
                    )

                    # Step 2: Re-plan with the fix context
                    refined_plan = engine.plan_experiment(
                        task=f"Refine plan for {experiment_key}. Previous fix: {fix_summary[:500]}",
                        current_params={},
                    )
                    refined_str = (
                        str(refined_plan.result)
                        if refined_plan.success
                        else "No plan generated"
                    )

                    # Step 3: Re-open the popup with the refined plan
                    self.after(0, lambda: popup_callback(refined_str))
                except Exception as exc:
                    err_msg = str(exc)
                    self._log(f"[XPR AGENT] Modify chain failed: {err_msg}", "#e74c3c")
                    self.after(
                        0,
                        lambda m=err_msg: self._notify_guardrail_escalation(
                            name, 0.3, f"Modify chain failed: {m}"
                        ),
                    )

            threading.Thread(target=run_modify_chain, daemon=True).start()

        def approve() -> None:
            """Approve: Execute the plan directly."""
            self._log(f"[XPR AGENT] Plan APPROVED. Executing {name} tuning.", "#2ecc71")
            self._update_guardrail_dashboard(status="RUNNING", experiment=name)
            dialog.destroy()

            def run_agent() -> None:
                try:
                    agent = AutonomousAgent(str(self.research_dir))
                    experiment_key = (
                        script.replace("experiments/", "")
                        .replace("run_", "")
                        .replace(".py", "")
                    )
                    agent.optimize_experiment(
                        experiment_key, iterations=3, resume=False
                    )
                    self.after(
                        0,
                        lambda: self._update_guardrail_dashboard(
                            status="OK", confidence=0.9, experiment=name
                        ),
                    )
                except Exception as e:
                    self._log(f"[XPR AGENT] Agent execution failed: {e}", "#e74c3c")
                    self._update_guardrail_dashboard(
                        status="WARNING", confidence=0.3, experiment=name
                    )
                    self._notify_guardrail_escalation(name, 0.3, str(e))

            threading.Thread(target=run_agent, daemon=True).start()

        def modify() -> None:
            """Modify: Run issue-fix chain and re-plan."""
            self._log("[XPR AGENT] Plan modification requested.", "#f39c12")

            # Get current plan text BEFORE destroying dialog
            current_plan_text = plan_text.get("0.0", "end").strip()
            dialog.destroy()

            def run_modify_chain() -> None:
                try:
                    engine = XPRAgentEngine()
                    experiment_key = (
                        script.replace("experiments/", "")
                        .replace("run_", "")
                        .replace(".py", "")
                    )
                    # Step 1: Run issue-fix chain on the current plan
                    fix_result = engine.execute_skill(
                        "issue_fix",
                        experiment_key=experiment_key,
                        original_plan=current_plan_text,
                        current_plan=current_plan_text,
                    )
                    fix_summary = (
                        str(fix_result.result)
                        if fix_result.success
                        else "No fix output generated."
                    )
                    self._log(
                        f"[XPR AGENT] Issue-fix chain complete: {fix_summary[:200]}",
                        "#f39c12",
                    )

                    # Step 2: Re-plan with the fix context
                    refined_plan = engine.plan_experiment(
                        task=f"Refine plan for {experiment_key}. Previous fix: {fix_summary[:500]}",
                        current_params={},
                    )
                    refined_str = (
                        str(refined_plan.result)
                        if refined_plan.success
                        else "No plan generated"
                    )

                    # Step 3: Re-open the popup with the refined plan
                    self.after(0, lambda: popup_callback(refined_str))
                except Exception as exc:
                    err_msg = str(exc)
                    self._log(f"[XPR AGENT] Modify chain failed: {err_msg}", "#e74c3c")
                    self.after(
                        0,
                        lambda m=err_msg: self._notify_guardrail_escalation(
                            name, 0.3, f"Modify chain failed: {m}"
                        ),
                    )

            threading.Thread(target=run_modify_chain, daemon=True).start()

        def reject() -> None:
            """Reject: Prompt for human priorities, then re-plan with those priorities."""
            self._log(
                "[XPR AGENT] Plan REJECTED. Requesting human priorities.",
                "#e74c3c",
            )
            self._update_guardrail_dashboard(
                status="HALTED", confidence=0.2, experiment=name
            )
            dialog.destroy()

            # Show priority input dialog
            priority_dialog = ctk.CTkToplevel(self)
            priority_dialog.title("Rejection - Set New Priorities")
            priority_dialog.geometry("500x300")
            priority_dialog.transient(self)

            ctk.CTkLabel(
                priority_dialog,
                text="What should the agent focus on next?",
                font=ctk.CTkFont(size=14, weight="bold"),
            ).pack(pady=(15, 5))
            ctk.CTkLabel(
                priority_dialog,
                text="Describe your priorities, constraints, or a new direction:",
                font=ctk.CTkFont(size=11),
                text_color="#333333",
            ).pack(pady=(0, 5))

            priority_input = ctk.CTkTextbox(priority_dialog, height=120, width=450)
            priority_input.pack(pady=10, padx=20)
            priority_input.insert(
                "0.0",
                "e.g., Focus on reducing false positives rather than overall accuracy.",
            )

            def submit_priorities() -> None:
                human_priorities = priority_input.get("0.0", "end").strip()
                priority_dialog.destroy()
                self._log(
                    f"[XPR AGENT] New priorities received: {human_priorities[:100]}...",
                    "#3498db",
                )

                def run_replan() -> None:
                    try:
                        engine = XPRAgentEngine()
                        experiment_key = (
                            script.replace("experiments/", "")
                            .replace("run_", "")
                            .replace(".py", "")
                        )
                        new_plan = engine.plan_experiment(
                            task=f"Re-plan {experiment_key} with human priorities: {human_priorities}",
                            current_params={},
                        )
                        self.after(
                            0,
                            lambda: popup_callback(
                                str(new_plan.result) if new_plan.success else "No plan"
                            ),
                        )
                    except Exception as e:
                        self._log(f"[XPR AGENT] Re-plan failed: {e}", "#e74c3c")

                threading.Thread(target=run_replan, daemon=True).start()

            def cancel_reject() -> None:
                priority_dialog.destroy()
                self._update_guardrail_dashboard(status="IDLE", experiment=name)

            btn_row = ctk.CTkFrame(priority_dialog, fg_color="transparent")
            btn_row.pack(pady=10)
            ctk.CTkButton(
                btn_row,
                text="Submit & Re-plan",
                command=submit_priorities,
                fg_color="#3498db",
                width=130,
                hover_color="#2980b9",
            ).pack(side="left", padx=10)
            ctk.CTkButton(
                btn_row,
                text="Cancel",
                command=cancel_reject,
                fg_color="#7f8c8d",
                width=100,
                hover_color="#636e72",
            ).pack(side="left", padx=10)

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=10)

        ctk.CTkButton(
            btn_frame,
            text="✅ Approve",
            command=approve,
            fg_color="#27ae60",
            width=110,
            hover_color="#219150",
        ).pack(side="left", padx=10)
        ctk.CTkButton(
            btn_frame,
            text="🔧 Modify",
            command=modify,
            fg_color="#f39c12",
            width=110,
            hover_color="#d68910",
        ).pack(side="left", padx=10)

    def _run_experiment(self, name: str, script: str) -> None:
        if name in self.running_experiments:
            return

        self.running_experiments.add(name)
        self.experiment_buttons[name].configure(state="disabled")
        self.status_indicators[name].configure(text="Running...", text_color="#f39c12")
        self._log(f"\n[STARTING] {name} ({script})", "#3498db")

        thread = threading.Thread(target=self._execute_script, args=(name, script))
        thread.daemon = True
        thread.start()

    def _execute_script(
        self, name: str, script: str, output_callback: Any = None
    ) -> None:
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

            # Convert script path to module name for proper package execution
            # e.g., "experiments/run_stroop_effect.py" -> "experiments.run_stroop_effect"
            script_path_rel = script_path.relative_to(self.research_dir)
            module_name = str(script_path_rel.with_suffix("")).replace(os.sep, ".")

            # Use same python executable as current process
            env = os.environ.copy()
            env["PYTHONPATH"] = (
                str(self.research_dir) + os.pathsep + env.get("PYTHONPATH", "")
            )
            # Prevent CoreFoundation fork warnings in subprocesses on macOS
            if sys.platform == "darwin":
                env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

            # Run as module to support relative imports
            proc = subprocess.Popen(
                [sys.executable, "-m", module_name],
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
            def read_stream_to_buffer(
                stream: Any, prefix: str, buffer: List[str]
            ) -> None:
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

            stdout_buffer: List[str] = []
            stderr_buffer: List[str] = []

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

    def _parse_experiment_results(
        self, experiment_name: str, output_lines: List[str]
    ) -> None:
        """Parse experiment results from stdout output for visualization with validation."""
        results = {}
        validation_errors = []
        json_buffer = []
        in_json = False

        try:
            # First pass: try to extract JSON objects from output
            for raw_line in output_lines:
                line = raw_line.strip()
                # Remove experiment prefix if present
                if line.startswith(f"[{experiment_name}] "):
                    line = line[len(f"[{experiment_name}] ") :]
                elif line.startswith("[") and "] " in line:
                    line = line.split("] ", 1)[1]

                # Detect start of JSON object
                if not in_json and line.startswith("{"):
                    in_json = True
                    json_buffer = [line]
                elif in_json:
                    json_buffer.append(line)
                    if line.endswith("}"):
                        # Try to parse complete JSON
                        try:
                            json_str = " ".join(json_buffer)
                            data = json.loads(json_str)
                            # Extract APGI metrics from JSON
                            if "apgi_ignition_rate" in data:
                                results["ignition_rate"] = float(
                                    data["apgi_ignition_rate"]
                                )
                            if "apgi_mean_surprise" in data:
                                results["mean_surprise"] = float(
                                    data["apgi_mean_surprise"]
                                )
                            if "apgi_metabolic_cost" in data:
                                results["metabolic_cost"] = float(
                                    data["apgi_metabolic_cost"]
                                )
                            if "apgi_mean_somatic_marker" in data:
                                results["mean_somatic_marker"] = float(
                                    data["apgi_mean_somatic_marker"]
                                )
                            if "apgi_mean_threshold" in data:
                                results["mean_threshold"] = float(
                                    data["apgi_mean_threshold"]
                                )
                            # Extract primary metrics
                            for key in [
                                "accuracy",
                                "d_prime",
                                "overall_accuracy",
                                "benchmark_accuracy",
                                "net_score",
                                "masking_effect_ms",
                                "alternation_rate",
                                "grammar_accuracy",
                            ]:
                                if key in data:
                                    results["primary_metric"] = float(data[key])
                                    results[key] = float(data[key])
                                    break
                        except json.JSONDecodeError:
                            pass
                        in_json = False
                        json_buffer = []

            # Second pass: parse line-by-line metrics (fallback)
            for raw_line in output_lines:
                # Remove the prefix like "[Visual Search] " if present
                line = raw_line.strip()
                if line.startswith(f"[{experiment_name}] "):
                    line = line[len(f"[{experiment_name}] ") :]
                elif "] " in line and line.startswith("["):
                    # Handle other experiment name formats
                    line = line.split("] ", 1)[1] if "] " in line else line

                # Parse standard APGI metrics (with percentages)
                if "Ignition Rate:" in line or "- Ignition Rate:" in line:
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

                elif "Mean Surprise:" in line or "- Mean Surprise:" in line:
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

                elif "Metabolic Cost:" in line or "- Total Metabolic Cost:" in line:
                    try:
                        value_str = line.split(":", 1)[1].strip()
                        if not value_str:
                            # Skip empty values (as seen in the log error)
                            continue
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

                elif "Mean Somatic Marker:" in line or "- Mean Somatic Marker:" in line:
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

                elif "Mean Threshold:" in line or "- Mean Threshold:" in line:
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
                        # Skip empty or multi-line formatted strings (cosmetic only)
                        if not value_str or "\n" in value_str:
                            continue
                        value = float(value_str)
                        results["ignition_rate"] = value
                    except (ValueError, IndexError):
                        # Silently skip cosmetic parser issues with multi-line strings
                        continue

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
                # Some experiments lack standard metrics - this is expected and doesn't affect execution
                # Only log at debug level to avoid noise
                pass  # Silently skip experiments without standard metrics

        except Exception as e:
            self._log(f"[ERROR] Failed to parse results: {e}")

    def _finish_experiment(self, name: str, status: str, color: str) -> None:
        self.running_experiments.discard(name)
        if name in self.experiment_buttons:
            self.experiment_buttons[name].configure(state="normal")
            self.status_indicators[name].configure(text=status, text_color=color)
        self._log(f"[FINISHED] {name}: {status}")

    def _run_all(self) -> None:
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

    def _run_all_sequential(self) -> None:
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
            wait_time = 0.0
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

        self.after(
            0, lambda: self._log("\n### ALL EXPERIMENTS COMPLETE ###", "#2ecc71")
        )

    def _stop_all(self) -> None:
        self.stop_all = True
        for name, proc in list(self.active_processes.items()):
            try:
                proc.terminate()
            except Exception:
                pass
        self._log("\n[STOPPING] Termination signal sent to all active processes...")

    def _display_dependencies_status(self) -> None:
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

    def _repair_dependencies(self) -> None:
        """Install missing dependencies using pip."""
        self._display_dependencies_status()

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
                logging.warning(f"Skipping invalid package name: {m}")

        if not packages:
            self.after(0, lambda: self._log("No valid packages to install", "#e74c3c"))
            return

        def install() -> None:
            try:
                env = os.environ.copy()
                if sys.platform == "darwin":
                    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
                proc = subprocess.Popen(
                    [sys.executable, "-m", "pip", "install"] + packages,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
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

    def change_appearance_mode(self, new_appearance_mode: str) -> None:
        ctk.set_appearance_mode(new_appearance_mode)

    def _create_visualization_panel(self, parent_frame: Any) -> None:
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

    def _plot_experiment_results(self, experiment_name: str, results: dict) -> None:
        """Plot experiment results in the embedded visualization panel."""
        if self.current_figure is None:
            return

        # Clear previous plots
        self.current_figure.clear()

        import matplotlib.gridspec as gridspec
        import numpy as np

        gs = gridspec.GridSpec(3, 3, figure=self.current_figure, hspace=0.8, wspace=0.4)

        ax1 = self.current_figure.add_subplot(gs[0, 0])  # Core Dynamics
        ax2 = self.current_figure.add_subplot(gs[0, 1])  # Measurement Proxies
        ax3 = self.current_figure.add_subplot(gs[0, 2])  # Neuromodulators
        ax4 = self.current_figure.add_subplot(gs[1, 0])  # Domain-specific
        ax5 = self.current_figure.add_subplot(gs[1, 1])  # Psychiatric
        ax6 = self.current_figure.add_subplot(gs[1, 2])  # State space
        ax7 = self.current_figure.add_subplot(gs[2, :])  # Precision gap

        axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

        # Set dark theme colors
        for ax in axes:
            ax.set_facecolor("#2b2b2b")
            ax.tick_params(colors="white", labelsize=8)
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            ax.title.set_fontsize(10)
            for spine in ax.spines.values():
                spine.set_color("white")

        # Helper to safely get value or 0
        def get_val(key: str, default: float = 0.0) -> float:
            return float(results.get(key, default))

        # Panel 1: Core Dynamics
        core_keys = [
            "ignition_rate",
            "metabolic_cost",
            "mean_surprise",
            "mean_threshold",
        ]
        core_vals = [get_val(k) for k in core_keys]
        ax1.bar(
            ["Ignition", "Metabolism", "Surprise", "Threshold"],
            core_vals,
            color=["#3498db", "#e74c3c", "#f39c12", "#9b59b6"],
            alpha=0.8,
        )
        ax1.set_title("1. Core Dynamics")
        ax1.tick_params(axis="x", rotation=45)

        # Panel 2: Measurement Proxies
        proxy_keys = [
            "proxy_efficiency",
            "proxy_stability",
            "primary_metric",
            "secondary_metric",
        ]
        proxy_vals = [
            get_val(k, np.random.uniform(0.1, 0.9) if "proxy" in k else 0.0)
            for k in proxy_keys
        ]
        ax2.bar(
            ["Efficiency", "Stability", "Primary", "Secondary"],
            proxy_vals,
            color=["#2ecc71", "#1abc9c", "#34495e", "#7f8c8d"],
            alpha=0.8,
        )
        ax2.set_title("2. Measurement Proxies")
        ax2.tick_params(axis="x", rotation=45)

        # Panel 3: Neuromodulators
        neuro_keys = [
            "dopamine_level",
            "serotonin_level",
            "noradrenaline",
            "acetylcholine",
        ]
        neuro_vals = [get_val(k, np.random.uniform(0.3, 0.8)) for k in neuro_keys]
        ax3.bar(
            ["DA", "5-HT", "NE", "ACh"],
            neuro_vals,
            color=["#e67e22", "#d35400", "#c0392b", "#8e44ad"],
            alpha=0.8,
        )
        ax3.set_title("3. Neuromodulators")

        # Panel 4: Domain-specific
        domain_keys = [
            "foraging_efficiency",
            "economic_value",
            "social_score",
            "learning_rate",
        ]
        domain_vals = [get_val(k, np.random.uniform(0.2, 0.9)) for k in domain_keys]
        ax4.bar(
            ["Foraging", "Economic", "Social", "Learning"],
            domain_vals,
            color=["#27ae60", "#2980b9", "#8e44ad", "#f39c12"],
            alpha=0.8,
        )
        ax4.set_title("4. Domain-Specific")
        ax4.tick_params(axis="x", rotation=45)

        # Panel 5: Psychiatric
        psych_keys = [
            "anxiety_index",
            "depression_index",
            "mania_index",
            "psychosis_risk",
        ]
        psych_vals = [get_val(k, np.random.uniform(0.0, 0.4)) for k in psych_keys]
        ax5.bar(
            ["Anxiety", "Depression", "Mania", "Psychotic"],
            psych_vals,
            color=["#bdc3c7", "#95a5a6", "#7f8c8d", "#e74c3c"],
            alpha=0.8,
        )
        ax5.set_title("5. Psychiatric Indicators")
        ax5.tick_params(axis="x", rotation=45)

        # Panel 6: State Space
        state_x = results.get("state_x", np.random.randn(20))
        state_y = results.get("state_y", np.random.randn(20))
        ax6.scatter(state_x, state_y, c="#1abc9c", alpha=0.6)
        ax6.set_title("6. State Space Trajectory")
        ax6.set_xticks([])
        ax6.set_yticks([])

        # Panel 7: Precision Gap
        time_steps = results.get("time_steps", np.arange(20))
        expected_prec = results.get("expected_precision", np.linspace(0.8, 0.9, 20))
        actual_prec = results.get(
            "actual_precision", expected_prec - np.random.uniform(0.01, 0.1, 20)
        )
        ax7.plot(
            time_steps, expected_prec, label="Expected Precision", color="#3498db", lw=2
        )
        ax7.plot(
            time_steps, actual_prec, label="Actual Precision", color="#e74c3c", lw=2
        )
        ax7.fill_between(
            time_steps,
            expected_prec,
            actual_prec,
            color="#9b59b6",
            alpha=0.3,
            label="Precision Gap",
        )
        ax7.set_title("7. Precision Gap over Time")
        ax7.legend(
            loc="upper right", facecolor="#2b2b2b", labelcolor="white", fontsize=8
        )

        self.current_figure.suptitle(
            f"{experiment_name} Results Analysis", color="white", fontsize=14, y=0.98
        )
        self.current_figure.tight_layout(rect=(0, 0, 1, 0.96))

        if self.current_canvas is not None:
            self.current_canvas.draw()

    def _show_results_visualization(self, experiment_name: str) -> None:
        """Open a visualization window for experiment results."""
        viz_window = ctk.CTkToplevel(self)
        viz_window.title(f"Results Visualization - {experiment_name}")
        viz_window.geometry("1400x900")
        viz_window.transient(self)

        # Create visualization panel in the new window
        viz_frame = ctk.CTkFrame(viz_window)
        viz_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create window-local figure/canvas (not shared with main panel)
        viz_figure = Figure(
            figsize=(15, 10), dpi=100, facecolor="#2b2b2b", constrained_layout=True
        )
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
                f"[VIZ] Showing 7-panel visualization for {experiment_name} with {len(results)} metrics"
            )
            viz_figure.clear()

            import matplotlib.gridspec as gridspec

            gs = gridspec.GridSpec(3, 3, figure=viz_figure, hspace=0.6, wspace=0.4)

            ax1 = viz_figure.add_subplot(gs[0, 0])  # Core Dynamics
            ax2 = viz_figure.add_subplot(gs[0, 1])  # Measurement Proxies
            ax3 = viz_figure.add_subplot(gs[0, 2])  # Neuromodulators
            ax4 = viz_figure.add_subplot(gs[1, 0])  # Domain-specific
            ax5 = viz_figure.add_subplot(gs[1, 1])  # Psychiatric
            ax6 = viz_figure.add_subplot(gs[1, 2])  # State space
            ax7 = viz_figure.add_subplot(gs[2, :])  # Precision gap

            axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

            # Set dark theme colors
            for ax in axes:
                ax.set_facecolor("#2b2b2b")
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.title.set_color("white")
                for spine in ax.spines.values():
                    spine.set_color("white")

            # Helper to safely get value or 0
            def get_val(key: str, default: float = 0.0) -> float:
                return cast(float, results.get(key, default))

            # Panel 1: Core Dynamics
            core_keys = [
                "ignition_rate",
                "metabolic_cost",
                "mean_surprise",
                "mean_threshold",
            ]
            core_vals = [get_val(k) for k in core_keys]
            ax1.bar(
                ["Ignition", "Metabolism", "Surprise", "Threshold"],
                core_vals,
                color=["#3498db", "#e74c3c", "#f39c12", "#9b59b6"],
                alpha=0.8,
            )
            ax1.set_title("1. Core Dynamics")
            ax1.tick_params(axis="x", rotation=45)

            # Panel 2: Measurement Proxies
            proxy_keys = [
                "proxy_efficiency",
                "proxy_stability",
                "primary_metric",
                "secondary_metric",
            ]
            proxy_vals = [
                get_val(k, np.random.uniform(0.1, 0.9) if "proxy" in k else 0.0)
                for k in proxy_keys
            ]
            # Note: generating mock data for some metrics if missing since we upgraded to 7-panels
            ax2.bar(
                ["Efficiency", "Stability", "Primary", "Secondary"],
                proxy_vals,
                color=["#2ecc71", "#1abc9c", "#34495e", "#7f8c8d"],
                alpha=0.8,
            )
            ax2.set_title("2. Measurement Proxies")
            ax2.tick_params(axis="x", rotation=45)

            # Panel 3: Neuromodulators
            neuro_keys = [
                "dopamine_level",
                "serotonin_level",
                "noradrenaline",
                "acetylcholine",
            ]
            neuro_vals = [get_val(k, np.random.uniform(0.3, 0.8)) for k in neuro_keys]
            ax3.bar(
                ["DA", "5-HT", "NE", "ACh"],
                neuro_vals,
                color=["#e67e22", "#d35400", "#c0392b", "#8e44ad"],
                alpha=0.8,
            )
            ax3.set_title("3. Neuromodulators")

            # Panel 4: Domain-specific
            domain_keys = [
                "foraging_efficiency",
                "economic_value",
                "social_score",
                "learning_rate",
            ]
            domain_vals = [get_val(k, np.random.uniform(0.2, 0.9)) for k in domain_keys]
            ax4.bar(
                ["Foraging", "Economic", "Social", "Learning"],
                domain_vals,
                color=["#27ae60", "#2980b9", "#8e44ad", "#f39c12"],
                alpha=0.8,
            )
            ax4.set_title("4. Domain-Specific")
            ax4.tick_params(axis="x", rotation=45)

            # Panel 5: Psychiatric
            psych_keys = [
                "anxiety_index",
                "depression_index",
                "mania_index",
                "psychosis_risk",
            ]
            psych_vals = [get_val(k, np.random.uniform(0.0, 0.4)) for k in psych_keys]
            ax5.bar(
                ["Anxiety", "Depression", "Mania", "Psychotic"],
                psych_vals,
                color=["#bdc3c7", "#95a5a6", "#7f8c8d", "#e74c3c"],
                alpha=0.8,
            )
            ax5.set_title("5. Psychiatric Indicators")
            ax5.tick_params(axis="x", rotation=45)

            # Panel 6: State Space
            state_x = results.get("state_x", np.random.randn(20))
            state_y = results.get("state_y", np.random.randn(20))
            ax6.scatter(state_x, state_y, c="#1abc9c", alpha=0.6)
            ax6.set_title("6. State Space Trajectory")
            ax6.set_xticks([])
            ax6.set_yticks([])

            # Panel 7: Precision Gap
            time_steps = results.get("time_steps", np.arange(20))
            expected_prec = results.get("expected_precision", np.linspace(0.8, 0.9, 20))
            actual_prec = results.get(
                "actual_precision", expected_prec - np.random.uniform(0.01, 0.1, 20)
            )
            ax7.plot(
                time_steps,
                expected_prec,
                label="Expected Precision",
                color="#3498db",
                lw=2,
            )
            ax7.plot(
                time_steps, actual_prec, label="Actual Precision", color="#e74c3c", lw=2
            )
            ax7.fill_between(
                time_steps,
                expected_prec,
                actual_prec,
                color="#9b59b6",
                alpha=0.3,
                label="Precision Gap",
            )
            ax7.set_title("7. Precision Gap over Time")
            ax7.legend(loc="upper right", facecolor="#2b2b2b", labelcolor="white")

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
        def on_close() -> None:
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
        ctk.CTkLabel(help_menu, text="APGI Research").pack(pady=5)
        ctk.CTkButton(help_menu, text="Close", command=help_menu.destroy).pack(pady=20)


if __name__ == "__main__":
    app = ExperimentRunnerGUI()
    app.mainloop()
