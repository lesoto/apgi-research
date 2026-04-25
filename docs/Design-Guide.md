# APGI Design Guide

Design standards and visual specifications for APGI Research application.

---

## 1. Design Philosophy

### Core Principles

- **Research-First Interface**: UI optimized for scientific experiment management and data visualization
- **High Information Density**: Display multiple experiments, metrics, and status indicators simultaneously
- **Progressive Disclosure**: Advanced controls (guardrails, hypothesis board) available but not overwhelming
- **Dark Mode Default**: Reduced eye strain during long research sessions

### Design Goals

- Clarity over decoration — every element serves a research function
- Consistent color coding for experiment states and system health
- Immediate visual feedback for long-running processes
- Accessibility for extended use in laboratory environments

---

## 2. UI Framework

### Primary Framework: CustomTkinter

```python
import customtkinter as ctk

# Framework initialization
ctk.set_appearance_mode("Dark")  # Default: Dark, Options: Dark, Light, System
```

### Window Specifications

| Property | Value |
| ---------- | ------- |
| Default Size | 1400×900 pixels |
| Minimum Size | 1200×700 pixels |
| Grid Layout | 3×2 (Menu, Sidebar+Main, Console) |
| Resize Behavior | Main content expands, sidebar fixed |

### Grid Layout Structure

```text
┌─────────────────────────────────────────────────────────┐
│                    Menu Bar (row 0)                     │
├──────────────┬────────────────────────────────────────┤
│              │                                        │
│   Sidebar    │          Main Content Area           │
│   (fixed)    │           (expandable)               │
│              │                                        │
│   Row 1      │             Row 1                      │
│   Weight: 0  │            Weight: 3                   │
├──────────────┴────────────────────────────────────────┤
│                 Console Area (row 2)                  │
│                     Weight: 1                         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Color Palette

### Primary Colors

| Role | Hex | Usage |
| ------ | ----- | ------- |
| **Success/Running** | `#27ae60` | Run buttons, success states, positive metrics |
| **Success Hover** | `#219150` | Button hover states |
| **Info/Ready** | `#3498db` | Status indicators, visualization buttons |
| **Info Hover** | `#2980b9` | Button hover states |
| **Danger/Stop** | `#e74c3c` | Stop buttons, errors, escalations |
| **Danger Hover** | `#c0392b` | Button hover states |
| **Auto/AI** | `#8e44ad` | AI-powered auto-improvement features |
| **Auto Hover** | `#9b59b6` | Button hover states |

### Semantic Status Colors

| Status | Hex | Application |
| -------- | ----- | ------------- |
| **Healthy/OK** | `#2ecc71` | Guardrail confidence > 0.7, no escalations |
| **Warning** | `#f39c12` | Confidence 0.4–0.7, regression < 5% |
| **Critical** | `#e74c3c` | Confidence < 0.4, escalation tripped |
| **Running** | `#3498db` | Active experiment processes |
| **Idle** | `#aaaaaa` | Inactive or ready states |

### Background & Surface Colors

| Element | Hex | Usage |
| --------- | ----- | ------- |
| **Guardrail Panel** | `#1a1a2e` | Specialized dashboard panel background |
| **Hypothesis Board** | `#2c3e50` | Scientific management panel background |
| **Card Background** | Default ctk | Experiment cards (inherited from theme) |
| **Console** | Default ctk | Terminal output area |

### Color Logic for Dynamic Elements

```python
# Guardrail status colors
status_colors = {
    "IDLE": "#2ecc71",
    "RUNNING": "#3498db",
    "OK": "#2ecc71",
    "WARNING": "#f39c12",
    "ESCALATED": "#e74c3c",
    "HALTED": "#e74c3c",
}

# Confidence color logic
if confidence > 0.7:
    color = "#2ecc71"      # Green
elif confidence > 0.4:
    color = "#f39c12"      # Yellow
else:
    color = "#e74c3c"      # Red

# Regression color logic
reg_pct = abs(regression) * 100
if reg_pct < 2:
    color = "#2ecc71"      # Green
elif reg_pct < 5:
    color = "#f39c12"      # Yellow
else:
    color = "#e74c3c"      # Red
```

---

## 4. Typography

### Font System

CustomTkinter's `CTkFont` provides consistent typography:

```python
# Header / Title
ctk.CTkFont(size=24, weight="bold")

# Section Headers
ctk.CTkFont(size=16, weight="bold")

# Panel Titles
ctk.CTkFont(size=13, weight="bold")

# Labels
ctk.CTkFont(size=12)
ctk.CTkFont(size=11)

# Console / Monospace
("Courier", 13)

# Italic (Empty States)
ctk.CTkFont(size=12, slant="italic")
```

### Text Color Hierarchy

| Element | Color | Notes |
| --------- | ------- | ------- |
| Primary labels | Default ctk | `#e2e2e2` in dark mode |
| Secondary/muted | `#aaaaaa` | Status labels, hints |
| Accent headers | `#e2e2e2` | Panel titles with bold |
| Status text | Dynamic | Based on status color logic |

---

## 5. Component Patterns

### Buttons

#### Primary Action Buttons

```python
ctk.CTkButton(
    parent,
    text="▶ Run All Experiments",
    command=handler,
    height=40,
    font=ctk.CTkFont(size=14, weight="bold"),
    fg_color="#27ae60",
    hover_color="#219150",
)
```

#### Danger/Stop Buttons

```python
ctk.CTkButton(
    parent,
    text="⏹ Stop All",
    command=handler,
    height=40,
    font=ctk.CTkFont(size=14, weight="bold"),
    fg_color="#e74c3c",
    hover_color="#c0392b",
)
```

#### Compact Card Buttons

```python
ctk.CTkButton(
    parent,
    text="▶ RUN",
    command=handler,
    width=70,
    height=28,
    fg_color="#27ae60",
    hover_color="#219150",
)
```

### Cards

#### Experiment Card Layout

```text
┌─────────────────────────────────────┐
│ Experiment Name          [RUN] [XPR]│
│ 📄 run_script.py                    │
│ Status: Ready          [VIZ]        │
└─────────────────────────────────────┘
```

```python
card = ctk.CTkFrame(parent)
card.grid_columnconfigure(0, weight=1)

# Title
ctk.CTkLabel(card, text=name, font=ctk.CTkFont(size=16, weight="bold"))

# Script reference
ctk.CTkLabel(card, text=f"📄 {script}", font=ctk.CTkFont(size=12), text_color="gray")

# Status indicator
ctk.CTkLabel(card, text="Ready", font=ctk.CTkFont(size=12), text_color="#3498db")
```

### Panels / Dashboards

#### Guardrail Dashboard

- **Background**: `#1a1a2e` (dark navy)
- **Corner Radius**: 8 pixels
- **Indicators**: 4 metrics with labels
- **Status colors**: Dynamic based on thresholds

```python
guardrail_frame = ctk.CTkFrame(
    parent,
    corner_radius=8,
    fg_color="#1a1a2e"
)
```

#### Hypothesis Board

- **Background**: `#2c3e50` (slate blue)
- **Corner Radius**: 8 pixels
- **Content**: Scrollable list of active hypotheses

```python
hypothesis_frame = ctk.CTkFrame(
    parent,
    corner_radius=8,
    fg_color="#2c3e50"
)
```

### Scrollable Areas

```python
# Main experiments list
scrollable_frame = ctk.CTkScrollableFrame(
    parent,
    label_text=f"Research Experiments ({count})"
)
scrollable_frame.grid_columnconfigure((0, 1), weight=1)  # 2-column grid

# Hypothesis list
hypothesis_scrollable = ctk.CTkScrollableFrame(
    parent,
    label_text="Active Hypotheses"
)
```

### Dialogs / Modals

```python
dialog = ctk.CTkToplevel(parent)
dialog.title("🧪 Create New Hypothesis")
dialog.geometry("600x500")
dialog.transient(parent)
dialog.attributes("-topmost", True)
```

---

## 6. Layout Specifications

### Spacing Scale

| Token | Value | Usage |
| ------- | ------- | ------- |
| `xs` | 2–5 px | Tight internal spacing |
| `sm` | 10 px | Default padding |
| `md` | 15–20 px | Card padding, section gaps |
| `lg` | 20 px | Major section separation |

### Grid Configuration

```python
# Main window layout
self.grid_columnconfigure(0, weight=0)  # Sidebar: fixed
self.grid_columnconfigure(1, weight=1)  # Main: expandable
self.grid_rowconfigure(0, weight=0)   # Menu: fixed
self.grid_rowconfigure(1, weight=3)    # Content: 3x priority
self.grid_rowconfigure(2, weight=1)    # Console: 1x priority
```

### Standard Padding Patterns

```python
# Sidebar elements
padx=20, pady=10

# Card internal
padx=15, pady=(15, 5)  # Top heavier

# Dashboard panels
padx=10, pady=(10, 5)

# Dialog forms
padx=20, pady=10
```

---

## 7. Icons & Visual Language

### Icon Set (Unicode Emoji)

| Icon | Meaning | Usage |
| ------ | ------- | ------- |
| ▶ | Run/Start | Primary action buttons |
| ⏹ | Stop/Halt | Termination actions |
| 🧹 | Clear/Clean | Maintenance actions |
| ⚡ | Guardrails | System safety features |
| 🧪 | Hypothesis | Scientific method features |
| ➕ | Add/Create | New item creation |
| 📋 | Review/List | View collections |
| 📄 | File/Script | Code references |
| 📊 | Visualize | Charts and graphs |
| 🤖 | AI/Auto | Machine learning features |
| ✅ | Success | Confirmation states |
| 🚨 | Alert | Error/escalation warnings |
| ⚠️ | Warning | Caution indicators |

### Status Indicators

Status indicators are labels with dynamic text colors:

```python
status_label = ctk.CTkLabel(
    card,
    text="Ready",           # "Ready" | "Running" | "Success" | "Failed"
    font=ctk.CTkFont(size=12),
    text_color="#3498db"    # Dynamic based on state
)
```

---

## 8. Visualization Integration

### Matplotlib Embedding

```python
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Create figure with dark theme
fig = Figure(figsize=(8, 4), dpi=100)
ax = fig.add_subplot(111)

# Embed in CustomTkinter
canvas = FigureCanvasTkAgg(fig, master=parent)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(row=0, column=0, sticky="nsew")
```

### Chart Color Scheme

Use the primary palette for chart elements:

- Green (`#27ae60`) for positive/success metrics
- Blue (`#3498db`) for neutral/data series
- Red (`#e74c3c`) for warnings/errors
- Purple (`#8e44ad`) for AI/ML predictions

---

## 9. Responsive Behavior

### Window Resize

| Breakpoint | Behavior |
| ---------- | -------- |
| >1400px | Full 2-column experiment grid |
| 1200–1400px | Maintain layout, reduce padding |
| <1200px | Single column experiment grid |

### Content Adaptation

```python
# Experiment cards arrange in 2 columns
row = index // 2
col = index % 2
```

---

## 10. Animation & Feedback

### Transitions

CustomTkinter handles button hover states automatically via `hover_color` parameter.

### Progress Indicators

- **Console logging**: Real-time text updates with color coding
- **Status labels**: Text changes with color transitions
- **Guardrail dashboard**: Live metric updates with color thresholds

---

## 11. Accessibility

### Contrast Requirements

- All text meets WCAG AA standards against dark backgrounds
- Status colors are distinguishable for colorblind users (shape + text + color)
- Interactive elements have clear hover states

### Keyboard Navigation

- Tab order follows visual layout
- Buttons have clear focus indicators (handled by CustomTkinter)
- Dialogs trap focus until dismissed

---

## 12. Implementation Example

### Complete Component Assembly

```python
import customtkinter as ctk

class ResearchDashboard(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        
        # Title
        title = ctk.CTkLabel(
            self,
            text="🔬 Active Experiments",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        
        # Action bar
        action_frame = ctk.CTkFrame(self, fg_color="transparent")
        action_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkButton(
            action_frame,
            text="▶ Run All",
            fg_color="#27ae60",
            hover_color="#219150",
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            action_frame,
            text="⏹ Stop All",
            fg_color="#e74c3c",
            hover_color="#c0392b",
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left", padx=5)

---

## 13. File References

| File | Design System Role |
|------|-------------------|
| `GUI_auto_improve_experiments.py` | Primary UI implementation, component patterns |
| `tests/test_gui_simple.py` | UI component validation tests |
| `pyproject.toml` | Dependency specification (CustomTkinter) |

---

*Design guide version: 1.0*
*Framework: CustomTkinter 5.0+*
*Appearance: Dark Mode (default)*
