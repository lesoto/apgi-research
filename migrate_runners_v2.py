#!/usr/bin/env python3
"""
Simple migration script to add standardized imports and cli_entrypoint to all run_*.py files.
"""

import re
from pathlib import Path

RUNNER_FILES = [
    "run_ai_benchmarking.py",
    "run_artificial_grammar_learning.py",
    "run_attentional_blink.py",
    "run_binocular_rivalry.py",
    "run_change_blindness.py",
    "run_change_blindness_full_apgi.py",
    "run_drm_false_memory.py",
    "run_dual_n_back.py",
    "run_eriksen_flanker.py",
    "run_go_no_go.py",
    "run_inattentional_blindness.py",
    "run_interoceptive_gating.py",
    "run_iowa_gambling_task.py",
    "run_masking.py",
    "run_metabolic_cost.py",
    "run_multisensory_integration.py",
    "run_navon_task.py",
    "run_posner_cueing.py",
    "run_probabilistic_category_learning.py",
    "run_serial_reaction_time.py",
    "run_simon_effect.py",
    "run_somatic_marker_priming.py",
    "run_sternberg_memory.py",
    "run_stop_signal.py",
    "run_stroop_effect.py",
    "run_time_estimation.py",
    "run_virtual_navigation.py",
    "run_visual_search.py",
    "run_working_memory_span.py",
]


def add_standardized_imports(content: str) -> str:
    """Add standardized imports if not already present."""
    if "from standard_apgi_runner import StandardAPGIRunner" in content:
        return content

    # Find the last import line
    lines = content.split("\n")
    import_end_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("from ") or line.startswith("import "):
            import_end_idx = i + 1

    # Add standardized imports
    new_imports = [
        "",
        "# Standardized APGI imports",
        "from standard_apgi_runner import StandardAPGIRunner",
        "from apgi_cli import cli_entrypoint, create_standard_parser",
    ]

    lines[import_end_idx:import_end_idx] = new_imports
    return "\n".join(lines)


def update_main_entrypoint(content: str, filename: str) -> str:
    """Update __main__ block to use cli_entrypoint."""
    # Extract experiment name
    experiment_name = filename.replace("run_", "").replace(".py", "_")

    # Remove existing __main__ block
    main_block_pattern = r'\nif __name__ == "__main__":.*'
    content = re.sub(main_block_pattern, "", content, flags=re.DOTALL)

    # Remove existing main() function if it's at the end
    # Keep it if it has other uses
    if content.strip().endswith("def main():"):
        # Remove the simple main function
        content = re.sub(
            r"\ndef main\(\):.*?\n    return results\n", "", content, flags=re.DOTALL
        )

    # Add standardized entry point
    class_name = experiment_name.replace("_", " ").title().replace(" ", "")
    experiment_title = experiment_name.replace("_", " ").title()
    entry_point = """

def main(args):
    \"\"\"Main function for running the experiment.\"\"\"
    runner = Enhanced{class_name}Runner()
    results = runner.run_experiment()
    return results


if __name__ == "__main__":
    parser = create_standard_parser("Run {experiment_title} experiment")
    cli_entrypoint(main, parser)
""".format(class_name=class_name, experiment_title=experiment_title)

    content = content.rstrip() + entry_point
    return content


def migrate_file(filepath: Path) -> bool:
    """Migrate a single file."""
    try:
        with open(filepath, "r") as f:
            content = f.read()

        original = content
        content = add_standardized_imports(content)
        content = update_main_entrypoint(content, filepath.name)

        if content != original:
            with open(filepath, "w") as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main() -> None:
    base_dir = Path(__file__).parent
    print("Migrating run_*.py files to use standardized patterns...")
    print("=" * 60)

    migrated = 0
    skipped = 0

    for filename in RUNNER_FILES:
        filepath = base_dir / filename
        if not filepath.exists():
            print(f"  SKIP: {filename} (not found)")
            skipped += 1
            continue

        print(f"  Processing: {filename}")
        if migrate_file(filepath):
            print("    ✓ Migrated")
            migrated += 1
        else:
            print("    - No changes needed")
            skipped += 1

    print("=" * 60)
    print(f"Migrated: {migrated}")
    print(f"Skipped: {skipped}")
    print(f"Total: {len(RUNNER_FILES)}")


if __name__ == "__main__":
    main()
