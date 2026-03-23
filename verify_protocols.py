#!/usr/bin/env python3
"""
APGI Experiment Protocol Verification Script

Verifies all experiment protocols against the 6 verification criteria:
1. File Structure: Both prepare_*.py and run_*.py files exist
2. READ-ONLY Designation: All prepare files properly marked as READ-ONLY
3. AGENT-EDITABLE Designation: All run files properly marked as AGENT-EDITABLE
4. Time Budget: All use 600-second TIME_BUDGET
5. Primary Metrics: All define and output primary metrics correctly
6. Import Structure: All run files properly import from their prepare files
"""

import re
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

# Experiment list from run_experiments.py and USAGE.md
EXPERIMENTS = [
    "ai_benchmarking",
    "artificial_grammar_learning",
    "attentional_blink",
    "binocular_rivalry",
    "change_blindness",
    "change_blindness_full_apgi",  # Full APGI variant of change_blindness
    "drm_false_memory",
    "dual_n_back",
    "eriksen_flanker",
    "go_no_go",
    "inattentional_blindness",
    "interoceptive_gating",
    "iowa_gambling_task",
    "masking",
    "metabolic_cost",
    "multisensory_integration",
    "navon_task",
    "posner_cueing",
    "probabilistic_category_learning",
    "serial_reaction_time",
    "simon_effect",
    "somatic_marker_priming",
    "sternberg_memory",
    "stop_signal",
    "stroop_effect",
    "time_estimation",
    "virtual_navigation",
    "visual_search",
    "working_memory_span",
]

# Additional experiments from file listing
ADDITIONAL_EXPERIMENTS = [
    "igt",  # Short name for iowa_gambling_task
]


@dataclass
class VerificationResult:
    experiment: str
    passed: List[str]
    failed: List[str]
    warnings: List[str]


def check_read_only_marker(filepath: Path) -> Tuple[bool, str]:
    """Check if prepare file has READ-ONLY marker in docstring."""
    try:
        content = filepath.read_text()

        # Check for READ-ONLY in docstring
        docstring = ""
        if '"""' in content:
            docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
            if docstring_match:
                docstring = docstring_match.group(1)
        if "READ-ONLY" in docstring or "read-only" in docstring.lower():
            if "Do not modify" in docstring or "do not modify" in docstring.lower():
                return True, "✅ READ-ONLY marker present with 'Do not modify'"
            return True, "⚠️ READ-ONLY marker present but missing 'Do not modify'"

        return False, "❌ Missing READ-ONLY marker in docstring"
    except Exception as e:
        return False, f"❌ Error reading file: {e}"


def check_agent_editable_marker(filepath: Path) -> Tuple[bool, str]:
    """Check if run file has AGENT-EDITABLE marker in docstring."""
    try:
        content = filepath.read_text()

        # Check for AGENT-EDITABLE in docstring
        docstring = ""
        if '"""' in content:
            docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
            if docstring_match:
                docstring = docstring_match.group(1)
        if "AGENT-EDITABLE" in docstring or "agent-editable" in docstring.lower():
            return True, "✅ AGENT-EDITABLE marker present"
        if "Modify this file" in docstring or "modifiable" in docstring.lower():
            return True, "⚠️ Modifiable hint present but not AGENT-EDITABLE"

        return False, "❌ Missing AGENT-EDITABLE marker in docstring"
    except Exception as e:
        return False, f"❌ Error reading file: {e}"


def check_time_budget(filepath: Path, file_type: str) -> Tuple[bool, str]:
    """Check if file has TIME_BUDGET = 600."""
    try:
        content = filepath.read_text()

        # Look for TIME_BUDGET = 600
        time_budget_pattern = r"TIME_BUDGET\s*=\s*(\d+)"
        matches = re.findall(time_budget_pattern, content)

        if matches:
            for match in matches:
                if int(match) == 600:
                    return True, "✅ TIME_BUDGET = 600 seconds"
                else:
                    return False, "❌ TIME_BUDGET = " + match + " (expected 600)"

        # Check for time budget in comments
        if "600" in content and (
            "time" in content.lower() or "budget" in content.lower()
        ):
            return True, "⚠️ 600 seconds mentioned but not as TIME_BUDGET constant"

        # Check for assertion
        if "assert TIME_BUDGET == 600" in content:
            return True, "✅ TIME_BUDGET asserted to be 600"

        return False, "❌ Missing TIME_BUDGET = 600"
    except Exception as e:
        return False, f"❌ Error reading file: {e}"


def check_primary_metric(filepath: Path, experiment: str) -> Tuple[bool, str]:
    """Check if run file outputs primary metric."""
    try:
        content = filepath.read_text()

        # Expected primary metrics per experiment (from USAGE.md)
        expected_metrics = {
            "ai_benchmarking": "benchmark_accuracy",
            "artificial_grammar_learning": "grammar_accuracy",
            "attentional_blink": "blink_magnitude",
            "binocular_rivalry": "alternation_rate",
            "change_blindness": "detection_rate",
            "drm_false_memory": "accuracy",
            "eriksen_flanker": "flanker_effect_ms",
            "go_no_go": "d_prime",
            "inattentional_blindness": "accuracy",
            "interoceptive_gating": "gating_threshold",
            "iowa_gambling_task": "net_score",
            "igt": "net_score",
            "masking": "masking_effect_ms",
            "metabolic_cost": "metabolic_cost_ratio",
            "multisensory_integration": "multisensory_gain_ms",
            "navon_task": "global_advantage_ms",
            "posner_cueing": "validity_effect_ms",
            "probabilistic_category_learning": "learning_rate",
            "serial_reaction_time": "learning_effect_ms",
            "simon_effect": "simon_effect_ms",
            "somatic_marker_priming": "priming_effect_ms",
            "sternberg_memory": "search_slope_ms_per_item",
            "stop_signal": "ssrt_ms",
            "stroop_effect": "interference_effect_ms",
            "time_estimation": "mean_error_percent",
            "virtual_navigation": "path_efficiency",
            "visual_search": "conjunction_present_slope",
            "working_memory_span": "d_prime",
        }

        expected_metric = expected_metrics.get(experiment, "")

        # Check for primary metric output
        if "primary_metric:" in content:
            return True, "✅ Outputs 'primary_metric:' format"

        # Check for expected metric name in output
        if expected_metric:
            if re.search(rf"{expected_metric}[:\s]", content):
                return True, "✅ Outputs expected metric: " + expected_metric

        # Check for common metric names
        common_metrics = [
            "net_score",
            "accuracy",
            "d_prime",
            "learning_rate",
            "interference_effect_ms",
        ]
        for metric in common_metrics:
            if metric in content and f"{metric}:" in content:
                return True, "⚠️ Outputs metric " + metric + " (may not match expected)"

        return (
            False,
            "❌ Missing primary metric output (expected: " + expected_metric + ")",
        )
    except Exception as e:
        return False, f"❌ Error reading file: {e}"


def check_import_structure(filepath: Path, experiment: str) -> Tuple[bool, str]:
    """Check if run file properly imports from prepare file."""
    try:
        content = filepath.read_text()

        # Determine prepare module name
        if experiment == "igt":
            prepare_module = "prepare_igt"
        else:
            prepare_module = f"prepare_{experiment}"

        # Check for import from prepare module
        import_patterns = [
            rf"from\s+{prepare_module}\s+import",
            rf"import\s+{prepare_module}",
        ]

        for pattern in import_patterns:
            if re.search(pattern, content):
                # Check for TIME_BUDGET import specifically
                if "TIME_BUDGET" in content:
                    return (
                        True,
                        "✅ Imports from " + prepare_module + " including TIME_BUDGET",
                    )
                return True, "✅ Imports from " + prepare_module

        return False, "❌ Missing import from " + prepare_module
    except Exception as e:
        return False, f"❌ Error reading file: {e}"


def verify_experiment(experiment: str, base_dir: Path) -> VerificationResult:
    """Verify a single experiment protocol."""
    passed = []
    failed = []
    warnings: list[str] = []

    # Determine file names
    if experiment == "igt":
        prepare_file = base_dir / "prepare_igt.py"
        run_file = base_dir / "run_igt.py"
    else:
        prepare_file = base_dir / f"prepare_{experiment}.py"
        run_file = base_dir / f"run_{experiment}.py"

    # Criterion 1: File Structure
    prepare_exists = prepare_file.exists()
    run_exists = run_file.exists()

    if prepare_exists and run_exists:
        passed.append("1. File Structure: Both files exist")
    else:
        if not prepare_exists:
            failed.append(f"1. File Structure: Missing {prepare_file.name}")
        if not run_exists:
            failed.append(f"1. File Structure: Missing {run_file.name}")
        return VerificationResult(experiment, passed, failed, warnings)

    # Criterion 2: READ-ONLY Designation (prepare file)
    ok, msg = check_read_only_marker(prepare_file)
    if ok:
        if "✅" in msg:
            passed.append(f"2. READ-ONLY: {msg}")
        else:
            warnings.append(f"2. READ-ONLY: {msg}")
    else:
        failed.append(f"2. READ-ONLY: {msg}")

    # Criterion 3: AGENT-EDITABLE Designation (run file)
    ok, msg = check_agent_editable_marker(run_file)
    if ok:
        if "✅" in msg:
            passed.append(f"3. AGENT-EDITABLE: {msg}")
        else:
            warnings.append(f"3. AGENT-EDITABLE: {msg}")
    else:
        failed.append(f"3. AGENT-EDITABLE: {msg}")

    # Criterion 4: Time Budget (both files)
    ok, msg = check_time_budget(prepare_file, "prepare")
    if ok:
        if "✅" in msg:
            passed.append(f"4. Time Budget (prepare): {msg}")
        else:
            warnings.append(f"4. Time Budget (prepare): {msg}")
    else:
        failed.append(f"4. Time Budget (prepare): {msg}")

    ok, msg = check_time_budget(run_file, "run")
    if ok:
        if "✅" in msg:
            passed.append(f"4. Time Budget (run): {msg}")
        else:
            warnings.append(f"4. Time Budget (run): {msg}")
    else:
        failed.append(f"4. Time Budget (run): {msg}")

    # Criterion 5: Primary Metrics (run file)
    ok, msg = check_primary_metric(run_file, experiment)
    if ok:
        if "✅" in msg:
            passed.append(f"5. Primary Metric: {msg}")
        else:
            warnings.append(f"5. Primary Metric: {msg}")
    else:
        failed.append(f"5. Primary Metric: {msg}")

    # Criterion 6: Import Structure (run file)
    ok, msg = check_import_structure(run_file, experiment)
    if ok:
        if "✅" in msg:
            passed.append(f"6. Import Structure: {msg}")
        else:
            warnings.append(f"6. Import Structure: {msg}")
    else:
        failed.append(f"6. Import Structure: {msg}")

    return VerificationResult(experiment, passed, failed, warnings)


def main():
    """Run verification on all experiments."""
    base_dir = Path(__file__).parent

    print("=" * 80)
    print("APGI Experiment Protocol Verification Report")
    print("=" * 80)
    print()

    results = []
    all_experiments = sorted(set(EXPERIMENTS + ADDITIONAL_EXPERIMENTS))

    for experiment in all_experiments:
        result = verify_experiment(experiment, base_dir)
        results.append(result)

    # Print detailed results
    for result in results:
        print(f"\n{'=' * 80}")
        print(f"Experiment: {result.experiment}")
        print("=" * 80)

        if result.passed:
            print("\n✅ PASSED:")
            for item in result.passed:
                print(f"  {item}")

        if result.warnings:
            print("\n⚠️  WARNINGS:")
            for item in result.warnings:
                print(f"  {item}")

        if result.failed:
            print("\n❌ FAILED:")
            for item in result.failed:
                print(f"  {item}")

        if not result.passed and not result.failed and not result.warnings:
            print("  (No files found)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total = len(results)
    complete = sum(1 for r in results if len(r.passed) >= 5 and len(r.failed) == 0)
    partial = sum(
        1
        for r in results
        if (len(r.passed) >= 3 and len(r.failed) <= 2)
        and not (len(r.passed) >= 5 and len(r.failed) == 0)
    )
    incomplete = total - complete - partial

    print("\nTotal experiments: " + str(total))
    print(
        "Complete protocols (all criteria): "
        + str(complete)
        + "/"
        + str(total)
        + " ("
        + str(100 * complete / total)[:4]
        + "%)"
    )
    print(
        "Partial protocols (most criteria): "
        + str(partial)
        + "/"
        + str(total)
        + " ("
        + str(100 * partial / total)[:4]
        + "%)"
    )
    print(
        "Incomplete protocols: "
        + str(incomplete)
        + "/"
        + str(total)
        + " ("
        + str(100 * incomplete / total)[:4]
        + "%)"
    )

    # List incomplete experiments
    incomplete_experiments = [
        r.experiment for r in results if len(r.passed) < 5 or len(r.failed) > 0
    ]
    if incomplete_experiments:
        print("\nExperiments needing attention:")
        for exp in incomplete_experiments:
            result = next(r for r in results if r.experiment == exp)
            print(
                f"  - {exp}: {len(result.failed)} failed, {len(result.warnings)} warnings"
            )

    # Detailed list of all experiments
    print(f"\n{'=' * 80}")
    print("COMPLETE EXPERIMENT LIST")
    print(f"{'=' * 80}")

    for result in sorted(results, key=lambda x: x.experiment):
        status = (
            "✅"
            if len(result.passed) >= 5 and len(result.failed) == 0
            else "⚠️"
            if len(result.passed) >= 3
            else "❌"
        )
        print(
            f"{status} {result.experiment:35} ({len(result.passed)} passed, {len(result.failed)} failed, {len(result.warnings)} warnings)"
        )

    return results


if __name__ == "__main__":
    results = main()
