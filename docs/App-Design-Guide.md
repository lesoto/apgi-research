# APGI Application Design Guide

**Autonomous Pretraining Research Interface**  
*Visual standards and component specifications for distribution*

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [UI Framework](#2-ui-framework)
3. [Color Palette](#3-color-palette)
4. [Typography](#4-typography)
5. [Icon System](#5-icon-system)
6. [Component Library](#6-component-library)
7. [Layout Specifications](#7-layout-specifications)
8. [Implementation Reference](#8-implementation-reference)

---

## 1. Design Philosophy

### Purpose

The APGI interface is designed for **research scientists managing psychological experiments**. The visual system prioritizes:

- **Clarity** — Information hierarchy is immediately apparent
- **Efficiency** — High-density layouts for complex experiment management
- **Accessibility** — WCAG 2.1 AA compliant for extended laboratory use
- **Professional Aesthetic** — Clean, credible presentation for academic distribution

### Interface Principles

| Principle | Implementation |
| --------- | -------------- |
| Progressive Disclosure | Advanced controls (guardrails, hypothesis board) nested but accessible |
| Status Visibility | System state visible at a glance via color-coded indicators |
| Consistent Feedback | Immediate visual response to all user actions |
| Research Context | Every element serves experiment management or data analysis |

---

## 2. UI Framework

### Technology Stack

| Component | Technology | Version |
| --------- | ---------- | ------- |
| UI Framework | CustomTkinter | ≥5.0.0 |
| Visualization | Matplotlib | ≥3.10 |
| Backend | Python | ≥3.10 |

### Window Specifications

```ascii
┌─────────────────────────────────────────────────────────────┐
│  Default Window Size: 1400 × 900 pixels                     │
│  Minimum Window Size: 1200 × 700 pixels                     │
│  Grid System: 3 rows × 2 columns                            │
└─────────────────────────────────────────────────────────────┘
```

### Grid Layout Architecture

```ascii
┌─────────────────────────────────────────────────────────────┐
│                    MENU BAR (fixed height)                    │
├────────────────┬──────────────────────────────────────────────┤
│                │                                              │
│   SIDEBAR      │          MAIN CONTENT AREA                 │
│   250px fixed  │          (expandable)                      │
│                │                                              │
├────────────────┴──────────────────────────────────────────────┤
│                 CONSOLE AREA (1× height)                     │
└─────────────────────────────────────────────────────────────┘
```

### Appearance Configuration

```python
import customtkinter as ctk

# Light mode default for distribution
ectk.set_appearance_mode("Light")
```

---

## 3. Color Palette

### Primary Brand Colors

| Color Name | Hex Code | Usage | Icon Representation |
| ---------- | -------- | ----- | ------------------- |
| **Success Green** | `#1e8449` | Run actions, positive metrics, healthy status | ![Check](https://api.iconify.design/lucide/check.svg?color=%231e8449) |
| **Info Blue** | `#2874a6` | Ready states, visualization buttons, neutral actions | ![Info](https://api.iconify.design/lucide/info.svg?color=%232874a6) |
| **Alert Red** | `#c0392b` | Stop actions, errors, critical escalations | ![Alert](https://api.iconify.design/lucide/alert-triangle.svg?color=%23c0392b) |
| **AI Purple** | `#7d3c98` | Machine learning features, auto-improvement | ![Bot](https://api.iconify.design/lucide/bot.svg?color=%237d3c98) |
| **Warning Orange** | `#d68910` | Caution states, partial success | ![Warning](https://api.iconify.design/lucide/alert-circle.svg?color=%23d68910) |

### Neutral Palette

| Role | Hex Code | Usage |
| ---- | -------- | ----- |
| **Background** | `#ffffff` | Primary window background |
| **Surface** | `#f8f9fa` | Cards, panels, elevated surfaces |
| **Border** | `#dee2e6` | Dividers, card borders |
| **Text Primary** | `#212529` | Headlines, important content |
| **Text Secondary** | `#6c757d` | Labels, descriptions, metadata |
| **Text Muted** | `#adb5bd` | Disabled states, placeholders |

### Semantic Status Colors

| Status | Background | Text | Indicator |
| ------ | ---------- | ---- | --------- |
| **Healthy/OK** | `#d4edda` | `#155724` | ![Check](https://api.iconify.design/lucide/check-circle.svg?color=%23155724) |
| **Running** | `#cce5ff` | `#004085` | ![Play](https://api.iconify.design/lucide/play.svg?color=%23004085) |
| **Warning** | `#fff3cd` | `#856404` | ![Alert](https://api.iconify.design/lucide/alert-triangle.svg?color=%23856404) |
| **Critical** | `#f8d7da` | `#721c24` | ![X](https://api.iconify.design/lucide/x-circle.svg?color=%23721c24) |
| **Idle** | `#e9ecef` | `#495057` | ![Minus](https://api.iconify.design/lucide/minus-circle.svg?color=%23495057) |

### Color Application Examples

```python
# Button styling (Light mode)
run_button = ctk.CTkButton(
    parent,
    text="Run All Experiments",
    fg_color="#1e8449",           # Success Green
    hover_color="#196f3d",        # Darker on hover
    text_color="#ffffff",         # White text
    height=40,
    font=ctk.CTkFont(size=14, weight="bold")
)

# Status indicator (Dynamic)
status_colors = {
    "IDLE":    ("#e9ecef", "#495057"),   # bg, text
    "RUNNING": ("#cce5ff", "#004085"),
    "OK":      ("#d4edda", "#155724"),
    "WARNING": ("#fff3cd", "#856404"),
    "ESCALATED": ("#f8d7da", "#721c24"),
    "HALTED":  ("#f8d7da", "#721c24"),
}
```

---

## 4. Typography

### Font Stack

CustomTkinter provides consistent cross-platform typography:

| Element | Font Specification | Size | Weight |
| ------- | ------------------ | ---- | ------ |
| **Application Title** | System Sans | 24px | Bold |
| **Section Header** | System Sans | 16px | Bold |
| **Panel Title** | System Sans | 13px | Bold |
| **Button Text** | System Sans | 14px | Bold |
| **Body Text** | System Sans | 12px | Normal |
| **Label/Caption** | System Sans | 11px | Normal |
| **Monospace** | Courier / Consolas | 13px | Normal |

### Code Reference

```python
# Header
ctk.CTkLabel(parent, text="APGI Research", font=ctk.CTkFont(size=24, weight="bold"))

# Section title
ctk.CTkLabel(parent, text="Active Experiments", font=ctk.CTkFont(size=16, weight="bold"))

# Body text
ctk.CTkLabel(parent, text="Experiment ready", font=ctk.CTkFont(size=12))

# Console / Log
ctk.CTkTextbox(parent, font=("Courier", 13))
```

---

## 5. Icon System

### Lucide Icon Set

The interface uses [Lucide](https://lucide.dev) icons for consistent, scalable vector graphics.

#### Action Icons

| Action | Icon Name | Preview | Usage |
| ------ | --------- | ------- | ----- |
| Run/Start | `play` | ![Play](https://api.iconify.design/lucide/play.svg) | Execute experiment |
| Stop/Halt | `square` | ![Square](https://api.iconify.design/lucide/square.svg) | Terminate process |
| Clear | `broom` | ![Broom](https://api.iconify.design/lucide/broom.svg) | Clear console |
| Add | `plus` | ![Plus](https://api.iconify.design/lucide/plus.svg) | Create new item |
| Review | `clipboard-list` | ![Clipboard](https://api.iconify.design/lucide/clipboard-list.svg) | View list |

#### Status Icons

| Status | Icon Name | Preview | Usage |
| ------ | --------- | ------- | ----- |
| Ready | `circle` | ![Circle](https://api.iconify.design/lucide/circle.svg) | Waiting state |
| Running | `loader-2` | ![Loader](https://api.iconify.design/lucide/loader-2.svg) | In progress |
| Success | `check-circle` | ![Check](https://api.iconify.design/lucide/check-circle.svg) | Completed |
| Failed | `x-circle` | ![X](https://api.iconify.design/lucide/x-circle.svg) | Error state |
| Warning | `alert-triangle` | ![Alert](https://api.iconify.design/lucide/alert-triangle.svg) | Caution |

#### Feature Icons

| Feature | Icon Name | Preview | Usage |
| ------- | --------- | ------- | ----- |
| Guardrails | `shield` | ![Shield](https://api.iconify.design/lucide/shield.svg) | Safety system |
| Hypothesis | `flask-conical` | ![Flask](https://api.iconify.design/lucide/flask-conical.svg) | Scientific method |
| Visualize | `bar-chart-3` | ![Chart](https://api.iconify.design/lucide/bar-chart-3.svg) | Charts/graphs |
| AI/Auto | `bot` | ![Bot](https://api.iconify.design/lucide/bot.svg) | ML features |
| File | `file-code` | ![File](https://api.iconify.design/lucide/file-code.svg) | Script reference |

### Icon Implementation

```python
# Using iconify CDN for documentation
icon_url = f"https://api.iconify.design/lucide/{icon_name}.svg?color={color}"

# In CustomTkinter (requires Pillow)
from PIL import Image, ImageTk
import requests
from io import BytesIO

def load_icon(icon_name, color="#212529", size=(20, 20)):
    url = f"https://api.iconify.design/lucide/{icon_name}.svg?color={color}"
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image = image.resize(size, Image.Resampling.LANCZOS)
    return ctk.CTkImage(light_image=image, size=size)
```

---

## 6. Component Library

### Primary Button

![Primary Button](https://api.iconify.design/lucide/play.svg?height=16&color=white) **Run All Experiments**

```python
ctk.CTkButton(
    parent,
    text="  Run All Experiments",
    image=play_icon,  # Optional icon
    compound="left",
    fg_color="#1e8449",
    hover_color="#196f3d",
    height=40,
    font=ctk.CTkFont(size=14, weight="bold"),
    corner_radius=6
)
```

**Specifications:**

- Height: 40px
- Corner radius: 6px
- Padding: 20px horizontal
- Icon gap: 8px (if using icon)

### Danger Button

![Danger](https://api.iconify.design/lucide/square.svg?height=16&color=white) **Stop All**

```python
ctk.CTkButton(
    parent,
    text="  Stop All",
    fg_color="#c0392b",
    hover_color="#a93226",
    height=40,
    font=ctk.CTkFont(size=14, weight="bold"),
    corner_radius=6
)
```

### Card Component

```text
┌───────────────────────────────────────────────────┐
│  Stroop Effect Experiment              [Run] [AI] │
│  📄 run_stroop_effect.py                          │
│  Status: ● Ready                      [View]      │
└───────────────────────────────────────────────────┘
```

```python
card = ctk.CTkFrame(
    parent,
    fg_color="#f8f9fa",
    corner_radius=8,
    border_width=1,
    border_color="#dee2e6"
)

# Title
ctk.CTkLabel(
    card,
    text="Experiment Name",
    font=ctk.CTkFont(size=16, weight="bold"),
    text_color="#212529"
)

# Script reference
ctk.CTkLabel(
    card,
    text="run_script.py",
    font=ctk.CTkFont(size=12),
    text_color="#6c757d"
)

# Status badge
status_badge = ctk.CTkLabel(
    card,
    text="● Ready",
    font=ctk.CTkFont(size=12),
    text_color="#2874a6"
)
```

### Guardrail Dashboard Panel

![Shield](https://api.iconify.design/lucide/shield.svg?height=24&color=%231e8449) **Guardrails**

```python
guardrail_frame = ctk.CTkFrame(
    parent,
    fg_color="#f8f9fa",
    corner_radius=8,
    border_width=1,
    border_color="#dee2e6"
)

# Panel header
ctk.CTkLabel(
    guardrail_frame,
    text="⚡ Guardrails",
    font=ctk.CTkFont(size=13, weight="bold"),
    text_color="#212529"
)

# Metric row
ctk.CTkLabel(parent, text="Status:", text_color="#6c757d", font=ctk.CTkFont(size=11))
ctk.CTkLabel(parent, text="IDLE", text_color="#1e8449", font=ctk.CTkFont(size=11, weight="bold"))
```

**Dashboard Metrics:**

- Status (IDLE, RUNNING, OK, WARNING, ESCALATED, HALTED)
- Confidence (0-1, color-coded)
- Regression (% change, color-coded)
- Escalation Count (0 = green, >0 = red)

### Hypothesis Board Panel

![Flask](https://api.iconify.design/lucide/flask-conical.svg?height=24&color=%232874a6) **Hypothesis Board**

```python
hypothesis_frame = ctk.CTkFrame(
    parent,
    fg_color="#f8f9fa",
    corner_radius=8,
    border_width=1,
    border_color="#dee2e6"
)
```

### Scrollable List

```python
scrollable = ctk.CTkScrollableFrame(
    parent,
    label_text="Research Experiments (24)",
    fg_color="#ffffff",
    border_width=1,
    border_color="#dee2e6"
)
```

### Console Area

```python
console = ctk.CTkTextbox(
    parent,
    font=("Courier", 13),
    fg_color="#f8f9fa",
    border_width=1,
    border_color="#dee2e6",
    text_color="#212529"
)
```

### Dialog/Modal

```python
dialog = ctk.CTkToplevel(parent)
dialog.title("Create New Hypothesis")
dialog.geometry("600x500")
dialog.transient(parent)

# Modal overlay effect
dialog.grab_set()
```

---

## 7. Layout Specifications

### Spacing Scale

| Token | Pixels | Usage |
| ----- | ------ | ----- |
| `xs` | 4px | Tight internal gaps |
| `sm` | 8px | Default component spacing |
| `md` | 16px | Section padding |
| `lg` | 24px | Major section separation |
| `xl` | 32px | Window edge padding |

### Grid Configuration Reference

```python
# Main window
self.grid_columnconfigure(0, weight=0, minsize=250)  # Sidebar
self.grid_columnconfigure(1, weight=1)                  # Main (expandable)
self.grid_rowconfigure(0, weight=0)                     # Menu
self.grid_rowconfigure(1, weight=3)                     # Content (priority)
self.grid_rowconfigure(2, weight=1)                     # Console
```

### Responsive Breakpoints

| Width | Layout Adjustment |
| ----- | ----------------- |
| ≥1400px | 2-column experiment grid |
| 1200–1400px | 2-column, reduced padding |
| <1200px | 1-column experiment stack |

---

## 8. Implementation Reference

### Complete Component Example

```python
import customtkinter as ctk

class ExperimentCard(ctk.CTkFrame):
    """Standard experiment card component."""
    
    def __init__(self, parent, name: str, script: str, status: str = "Ready"):
        super().__init__(
            parent,
            fg_color="#f8f9fa",
            corner_radius=8,
            border_width=1,
            border_color="#dee2e6"
        )
        
        self.grid_columnconfigure(0, weight=1)
        
        # Title
        self.title = ctk.CTkLabel(
            self,
            text=name,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#212529"
        )
        self.title.grid(row=0, column=0, padx=16, pady=(16, 4), sticky="w")
        
        # Script
        self.script = ctk.CTkLabel(
            self,
            text=f"📄 {script}",
            font=ctk.CTkFont(size=12),
            text_color="#6c757d"
        )
        self.script.grid(row=1, column=0, padx=16, pady=(0, 8), sticky="w")
        
        # Status
        status_colors = {
            "Ready": "#2874a6",
            "Running": "#004085",
            "Success": "#155724",
            "Failed": "#721c24"
        }
        self.status = ctk.CTkLabel(
            self,
            text=f"● {status}",
            font=ctk.CTkFont(size=12),
            text_color=status_colors.get(status, "#6c757d")
        )
        self.status.grid(row=2, column=0, padx=16, pady=(0, 16), sticky="w")
```

### File Structure

```text
apgi-research/
├── GUI_auto_improve_experiments.py   # Main UI implementation
├── app-design-guide.md               # This document
├── design-guide.md                   # Dark mode variant
├── pyproject.toml                    # Dependencies
└── tests/
    └── test_gui_simple.py            # UI validation tests
```

### Dependencies

```toml
[project.dependencies]
customtkinter = ">=5.0.0"
matplotlib = ">=3.10.8"
Pillow = ">=9.5.0"  # For icon support
```

---

## Appendix: Color Quick Reference

### Print-Ready Palette (CMYK Approximation)

| Hex      | CMYK                | Usage           |
|----------|---------------------|-----------------|
| `#1e8449`| C85 M35 Y90 K15     | Success Green   |
| `#2874a6`| C85 M45 Y15 K5      | Info Blue       |
| `#c0392b`| C15 M90 Y85 K10     | Alert Red       |
| `#7d3c98`| C60 M75 Y15 K5      | AI Purple       |
| `#212529`| C70 M60 Y55 K65     | Text Primary    |
| `#6c757d`| C55 M40 Y35 K15     | Text Secondary  |

---

**Document Version:** 1.0  
**Date:** April 2026  
**For:** APGI Research Platform Distribution
