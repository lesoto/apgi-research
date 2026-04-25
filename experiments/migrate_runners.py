#!/usr/bin/env python3
"""
Migration script to standardize all run_*.py files to use StandardAPGIRunner and cli_entrypoint.

This script:
1. Replaces custom APGI integration with StandardAPGIRunner
2. Updates entry points to use cli_entrypoint() pattern
3. Removes direct imports of APGIIntegration, APGIParameters
4. Adds standardized imports
"""

import re
from pathlib import Path
from typing import Tuple

# List of all run_*.py files to migrate
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


def extract_experiment_name(filename: str) -> str:
    """Extract experiment name from filename (e.g., run_stroop_effect.py -> stroop_effect)."""
    return filename.replace("run_", "").replace(".py", "")


def update_imports(content: str, experiment_name: str) -> str:
    """Update imports to use StandardAPGIRunner and cli_entrypoint."""

    # Remove old APGI imports
    old_imports_patterns = [
        r"from apgi_integration import.*APGIIntegration.*",
        r"from apgi_integration import.*APGIParameters.*",
        r"from apgi_integration import.*format_apgi_output.*",
        r"from ultimate_apgi_template import.*",
    ]

    for pattern in old_imports_patterns:
        content = re.sub(pattern, "# Removed: migrated to StandardAPGIRunner", content)

    # Add new standardized imports at the top after existing imports
    new_imports = """# Standardized APGI imports
from .standard_apgi_runner import StandardAPGIRunner
from apgi_cli import cli_entrypoint, create_standard_parser
"""

    # Find the last import line and add after it
    lines = content.split("\n")
    import_end_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("from ") or line.startswith("import "):
            import_end_idx = i + 1

    lines.insert(import_end_idx, new_imports)
    content = "\n".join(lines)

    return content


def update_runner_class(content: str, experiment_name: str) -> str:
    """Update the runner class to use StandardAPGIRunner."""

    # Find the main runner class (usually ends with "Runner")
    # Pattern: class SomeRunner:
    class_pattern = r"class (\w+Runner):"
    matches = list(re.finditer(class_pattern, content))

    if not matches:
        return content

    # Use the last match (likely the main runner class)
    main_class_match = matches[-1]
    class_name = main_class_match.group(1)

    # Add StandardAPGIRunner integration to __init__ if it exists
    init_pattern = rf"class {class_name}:(.*?)(?=def |\nclass |\Z)"
    init_match = re.search(init_pattern, content, re.DOTALL)

    if init_match:
        init_section = init_match.group(1)

        # Check if it already has APGI integration
        if "StandardAPGIRunner" not in init_section:
            # Add StandardAPGIRunner initialization
            # Find the __init__ method
            init_method_pattern = r"def __init__\(self.*?\):(.*?)(?=\n    def |\Z)"
            init_method_match = re.search(init_method_pattern, init_section, re.DOTALL)

            if init_method_match:
                init_body = init_method_match.group(1)

                # Add StandardAPGIRunner setup at the end of __init__
                apgi_setup = f"""
        # Initialize StandardAPGIRunner
        self.apgi_runner = StandardAPGIRunner(
            base_runner=self,
            experiment_name="{experiment_name}",
            enable_hierarchical=True,
            enable_precision_gap=True,
        )"""

                # Insert before the last line of __init__
                init_body = init_body.rstrip() + apgi_setup
                content = content.replace(init_method_match.group(1), init_body)

    return content


def update_main_entrypoint(content: str, experiment_name: str) -> str:
    """Update the __main__ block to use cli_entrypoint pattern."""

    # Remove existing __main__ block
    main_block_pattern = r'\nif __name__ == "__main__":.*'
    content = re.sub(main_block_pattern, "", content, flags=re.DOTALL)

    # Remove existing main() function if it exists after __main__
    # We'll add a new standardized main function

    # Add standardized main function and entry point
    class_name = experiment_name.replace("_", " ").title().replace(" ", "")
    standardized_main = """

def main(args):
    \"\"\"Main function for running the experiment.\"\"\"
    # Create and run the experiment
    runner = Enhanced{class_name}Runner()
    results = runner.run_experiment()

    # Print results
    print("Results for {experiment_name}:")
    for result_key, result_value in results.items():
        if result_key != "apgi_formatted":  # Skip formatted APGI output
            print("  {{}}: {{}}".format(result_key, result_value))

    # Print APGI metrics if available
    if results.get("apgi_enabled"):
        print("\\n" + results.get("apgi_formatted", ""))

    return results


if __name__ == "__main__":
    parser = create_standard_parser("Run {experiment_name} experiment")
    cli_entrypoint(main, parser)
""".format(class_name=class_name, experiment_name=experiment_name)

    content = content.rstrip() + standardized_main

    return content


def migrate_file(filepath: Path, experiment_name: str) -> Tuple[bool, str]:
    """Migrate a single runner file."""
    try:
        with open(filepath, "r") as f:
            content = f.read()

        original_content = content

        # Apply migrations
        content = update_imports(content, experiment_name)
        content = update_runner_class(content, experiment_name)
        content = update_main_entrypoint(content, experiment_name)

        # Write back if changed
        if content != original_content:
            with open(filepath, "w") as f:
                f.write(content)
            return True, "Successfully migrated"
        else:
            return False, "No changes needed"

    except Exception as e:
        return False, f"Error: {str(e)}"


def main() -> None:
    """Main migration function."""
    base_dir = Path(__file__).parent

    print("Starting migration of run_*.py files...")
    print("=" * 60)

    success_count = 0
    skip_count = 0
    error_count = 0

    for filename in RUNNER_FILES:
        filepath = base_dir / filename
        experiment_name = extract_experiment_name(filename)

        if not filepath.exists():
            print(f"⚠️  SKIP: {filename} (file not found)")
            skip_count += 1
            continue

        print(f"🔄 Processing: {filename} -> {experiment_name}")
        success, message = migrate_file(filepath, experiment_name)

        if success:
            print(f"  ✅ {message}")
            success_count += 1
        elif "No changes needed" in message:
            print(f"  ⏭️  {message}")
            skip_count += 1
        else:
            print(f"  ❌ {message}")
            error_count += 1

    print("=" * 60)
    print("Migration complete:")
    print(f"  Successfully migrated: {success_count}")
    print(f"  Skipped (no changes): {skip_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total files: {len(RUNNER_FILES)}")


if __name__ == "__main__":
    main()
