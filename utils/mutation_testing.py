"""
================================================================================
MUTATION TESTING CONFIGURATION
================================================================================

This module provides mutation testing configuration for verifying test
suite effectiveness through code mutation analysis.

Mutation testing introduces small changes (mutations) to the source code and
verifies that the test suite detects these changes (i.e., tests fail).

Features:
- Configurable mutation operators
- Target selection rules
- Weak assertion detection
- Mutation score calculation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MutationOperator:
    """Represents a mutation operator."""

    name: str
    description: str
    target_nodes: List[str]
    replacement_map: Dict[str, str]

    def apply(self, code: str) -> List[str]:
        """Apply mutation operator to code and return mutants."""
        mutants = []
        for target, replacement in self.replacement_map.items():
            mutant = code.replace(target, replacement)
            if mutant != code:
                mutants.append(mutant)
        return mutants


class MutationConfig:
    """Configuration for mutation testing."""

    # Standard mutation operators
    OPERATORS: List[MutationOperator] = [
        # Arithmetic operator mutation
        MutationOperator(
            name="AOR",  # Arithmetic Operator Replacement
            description="Replace arithmetic operators",
            target_nodes=["Add", "Sub", "Mult", "Div"],
            replacement_map={
                " + ": " - ",
                " - ": " + ",
                " * ": " / ",
                " / ": " * ",
            },
        ),
        # Comparison operator mutation
        MutationOperator(
            name="COR",  # Comparison Operator Replacement
            description="Replace comparison operators",
            target_nodes=["Eq", "NotEq", "Lt", "LtE", "Gt", "GtE"],
            replacement_map={
                " == ": " != ",
                " != ": " == ",
                " < ": " >= ",
                " > ": " <= ",
                " <= ": " > ",
                " >= ": " < ",
            },
        ),
        # Logical operator mutation
        MutationOperator(
            name="LOR",  # Logical Operator Replacement
            description="Replace logical operators",
            target_nodes=["And", "Or"],
            replacement_map={
                " and ": " or ",
                " or ": " and ",
            },
        ),
        # Constant replacement
        MutationOperator(
            name="CRR",  # Constant Replacement
            description="Replace numeric constants",
            target_nodes=["Constant"],
            replacement_map={
                " 0 ": " 1 ",
                " 1 ": " 0 ",
                " True ": " False ",
                " False ": " True ",
            },
        ),
    ]

    # Files to mutate
    TARGET_PATTERNS: List[str] = [
        "APGI_System.py",
        "base_experiment.py",
        "apgi_*.py",
    ]

    # Files to exclude from mutation
    EXCLUDE_PATTERNS: List[str] = [
        "*/tests/*",
        "*/test_*",
        "conftest.py",
        "*mutation*",
    ]

    # Mutation testing thresholds
    SURVIVED_THRESHOLD: int = 5  # Max acceptable surviving mutants
    TIMEOUT_SECONDS: int = 300  # Max time for mutation testing


@dataclass
class Mutant:
    """Represents a mutated version of code."""

    operator: str
    original_line: str
    mutated_line: str
    line_number: int
    file_path: Path
    status: str = "pending"  # pending, killed, survived, timeout


@dataclass
class MutationReport:
    """Report from mutation testing."""

    total_mutants: int = 0
    killed: int = 0
    survived: int = 0
    timeout: int = 0
    errors: int = 0
    duration: float = 0.0
    weak_assertions: List[Dict[str, Any]] = field(default_factory=list)
    surviving_mutants: List[Mutant] = field(default_factory=list)

    @property
    def mutation_score(self) -> float:
        """Calculate mutation score (percentage of mutants killed)."""
        if self.total_mutants == 0:
            return 100.0
        return (self.killed / self.total_mutants) * 100

    @property
    def is_acceptable(self) -> bool:
        """Check if mutation score is acceptable."""
        return (
            self.mutation_score >= 80.0
            and self.survived <= MutationConfig.SURVIVED_THRESHOLD
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "summary": {
                "total_mutants": self.total_mutants,
                "killed": self.killed,
                "survived": self.survived,
                "timeout": self.timeout,
                "errors": self.errors,
                "mutation_score": round(self.mutation_score, 2),
                "acceptable": self.is_acceptable,
                "duration_seconds": round(self.duration, 2),
            },
            "weak_assertions": self.weak_assertions,
            "surviving_mutants": [
                {
                    "operator": m.operator,
                    "file": str(m.file_path),
                    "line": m.line_number,
                    "original": m.original_line.strip(),
                    "mutated": m.mutated_line.strip(),
                }
                for m in self.surviving_mutants
            ],
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results."""
        recs = []

        if self.mutation_score < 80:
            recs.append(
                f"Improve mutation score from {self.mutation_score:.1f}% to 80%+"
            )

        if self.survived > MutationConfig.SURVIVED_THRESHOLD:
            recs.append(f"Add tests to kill {self.survived} surviving mutants")

        if self.weak_assertions:
            recs.append(f"Strengthen {len(self.weak_assertions)} weak assertions")

        if not recs:
            recs.append("Mutation testing results are good - maintain test quality")

        return recs


class WeakAssertionDetector:
    """Detect weak assertions that might not catch mutations."""

    WEAK_PATTERNS: List[str] = [
        "assert True",
        "assert False",
        "assert None",
        "assertEqual(x, x)",
        "assertIs(x, x)",
        "assertTrue(True)",
        "assertFalse(False)",
    ]

    @classmethod
    def detect(cls, test_file: Path) -> List[Dict[str, Any]]:
        """Detect weak assertions in test file."""
        weak_assertions = []

        try:
            content = test_file.read_text()
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                for pattern in cls.WEAK_PATTERNS:
                    if pattern in stripped:
                        weak_assertions.append(
                            {
                                "file": str(test_file),
                                "line": i,
                                "assertion": stripped,
                                "reason": "Tautological or vacuous assertion",
                            }
                        )
        except Exception as e:
            weak_assertions.append(
                {
                    "file": str(test_file),
                    "error": str(e),
                }
            )

        return weak_assertions


class MutationTester:
    """Perform mutation testing on code."""

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self.project_root = project_root or Path(".")
        self.report = MutationReport()

    def find_target_files(self) -> List[Path]:
        """Find files to mutate based on configuration."""
        targets = []

        for pattern in MutationConfig.TARGET_PATTERNS:
            for file_path in self.project_root.rglob(pattern):
                # Check exclusions
                excluded = any(
                    exclude in str(file_path)
                    for exclude in MutationConfig.EXCLUDE_PATTERNS
                )
                if not excluded:
                    targets.append(file_path)

        return list(set(targets))  # Remove duplicates

    def generate_mutants(self, file_path: Path) -> List[Mutant]:
        """Generate mutants for a file."""
        mutants = []

        try:
            content = file_path.read_text()
            lines = content.split("\n")

            for line_num, line in enumerate(lines, 1):
                for operator in MutationConfig.OPERATORS:
                    mutated_lines = operator.apply(line)
                    for mutated_line in mutated_lines:
                        if mutated_line != line:
                            mutant = Mutant(
                                operator=operator.name,
                                original_line=line,
                                mutated_line=mutated_line,
                                line_number=line_num,
                                file_path=file_path,
                            )
                            mutants.append(mutant)

        except Exception as e:
            print(f"Error generating mutants for {file_path}: {e}")

        return mutants

    def run_tests_against_mutant(self, mutant: Mutant) -> str:
        """Run tests against a mutant and return status."""
        import subprocess
        import tempfile

        # Create temp file with mutant
        try:
            original_content = mutant.file_path.read_text()
            lines = original_content.split("\n")
            lines[mutant.line_number - 1] = mutant.mutated_line
            mutated_content = "\n".join(lines)

            # Write mutated content
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as temp_file:
                temp_file.write(mutated_content)
                temp_path = Path(temp_file.name)

            # Run tests
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-x", "--tb=no"],
                capture_output=True,
                timeout=MutationConfig.TIMEOUT_SECONDS,
            )

            # Clean up
            temp_path.unlink(missing_ok=True)

            # Determine status
            if result.returncode != 0:
                return "killed"  # Tests failed = mutant killed
            else:
                return "survived"  # Tests passed = mutant survived

        except subprocess.TimeoutExpired:
            return "timeout"
        except Exception as e:
            print(f"Error testing mutant: {e}")
            return "error"

    def run_mutation_testing(self, max_mutants: Optional[int] = None) -> MutationReport:
        """Run full mutation testing suite."""
        import time

        print("Starting mutation testing...")
        start_time = time.time()

        # Find target files
        target_files = self.find_target_files()
        print(f"Found {len(target_files)} target files")

        # Generate mutants
        all_mutants: List[Mutant] = []
        for file_path in target_files:
            mutants = self.generate_mutants(file_path)
            all_mutants.extend(mutants)

        # Limit if specified
        if max_mutants:
            all_mutants = all_mutants[:max_mutants]

        self.report.total_mutants = len(all_mutants)
        print(f"Generated {len(all_mutants)} mutants")

        # Test each mutant
        for i, mutant in enumerate(all_mutants, 1):
            print(
                f"Testing mutant {i}/{len(all_mutants)}: {mutant.operator} at line {mutant.line_number}"
            )

            status = self.run_tests_against_mutant(mutant)
            mutant.status = status

            # Update counts
            if status == "killed":
                self.report.killed += 1
            elif status == "survived":
                self.report.survived += 1
                self.report.surviving_mutants.append(mutant)
            elif status == "timeout":
                self.report.timeout += 1
            else:
                self.report.errors += 1

        self.report.duration = time.time() - start_time

        # Detect weak assertions
        print("Checking for weak assertions...")
        for test_file in (self.project_root / "tests").glob("test_*.py"):
            weak = WeakAssertionDetector.detect(test_file)
            self.report.weak_assertions.extend(weak)

        return self.report

    def print_summary(self) -> None:
        """Print mutation testing summary."""
        print("\n" + "=" * 70)
        print("MUTATION TESTING SUMMARY")
        print("=" * 70)

        print(f"\nTotal Mutants: {self.report.total_mutants}")
        print(f"  Killed: {self.report.killed} ({self.report.mutation_score:.1f}%)")
        print(f"  Survived: {self.report.survived}")
        print(f"  Timeout: {self.report.timeout}")
        print(f"  Errors: {self.report.errors}")

        print(f"\nMutation Score: {self.report.mutation_score:.2f}%")
        print(f"Acceptable: {self.report.is_acceptable}")

        if self.report.weak_assertions:
            print(f"\nWeak Assertions Found: {len(self.report.weak_assertions)}")

        if self.report.surviving_mutants:
            print(f"\nSurviving Mutants: {len(self.report.surviving_mutants)}")
            for mutant in self.report.surviving_mutants[:5]:  # Show first 5
                print(
                    f"  - {mutant.operator} at {mutant.file_path}:{mutant.line_number}"
                )

        print("\n" + "=" * 70)


# =============================================================================
# MUTATION TESTING CLI
# =============================================================================


def main() -> int:
    """Run mutation testing from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run mutation testing")
    parser.add_argument(
        "--max-mutants",
        type=int,
        default=None,
        help="Maximum number of mutants to generate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_reports/mutation_report.json"),
        help="Output path for mutation report",
    )
    args = parser.parse_args()

    # Run mutation testing
    tester = MutationTester()
    report = tester.run_mutation_testing(max_mutants=args.max_mutants)

    # Print summary
    tester.print_summary()

    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"\nReport saved to: {args.output}")

    # Return exit code
    return 0 if report.is_acceptable else 1


if __name__ == "__main__":
    import json
    import sys

    sys.exit(main())
