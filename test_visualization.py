#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from APGI_System import APGIStateLibrary, CompleteAPGIVisualizer


def test_visualization():
    """Test the visualization part separately"""
    print("Testing visualization...")

    # Create library and visualizer
    library = APGIStateLibrary()
    visualizer = CompleteAPGIVisualizer(library)

    # Create mock history data
    import numpy as np

    history = {
        "time": np.linspace(0, 10, 100),
        "S": np.random.random(100),
        "theta": np.random.random(100),
        "B": np.random.random(100),
    }

    try:
        # Test the dashboard creation
        fig = visualizer.plot_comprehensive_dashboard(history)
        print(f"Figure created successfully: {type(fig)}")
        print(f"Figure has items method: {hasattr(fig, 'items')}")

        # Save the figure
        fig.savefig("test_dashboard.png", dpi=150, bbox_inches="tight")
        print("Figure saved successfully")

        plt.close(fig)
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    success = test_visualization()
    print(f"Test {'PASSED' if success else 'FAILED'}")
