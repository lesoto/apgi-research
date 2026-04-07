"""
================================================================================
TEST ORCHESTRATION RUNNER
================================================================================

This module provides comprehensive test orchestration including:
- Automated test discovery and execution
- Coverage report generation
- Structured output reports
- Test categorization and filtering
- Mutation testing integration
- Performance reporting
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TestResult:
    """Test result data structure."""

    name: str
    status: str
    duration: float
    message: str = ""
    category: str = ""
    file_path: Optional[str] = None

    @property
    def test_name(self) -> str:
        """Get test name for backward compatibility."""
        return self.name


@dataclass
class TestSuiteReport:
    """Complete test suite report."""

    timestamp: str
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration: float = 0.0
    coverage_percent: float = 0.0
    branch_coverage: float = 0.0
    results: List[TestResult] = field(default_factory=list)
    coverage_gaps: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total_tests": self.total_tests,
                "passed": self.passed,
                "failed": self.failed,
                "skipped": self.skipped,
                "errors": self.errors,
                "duration_seconds": round(self.duration, 2),
                "coverage_percent": round(self.coverage_percent, 2),
                "branch_coverage_percent": round(self.branch_coverage, 2),
            },
            "test_results": [
                {
                    "name": r.test_name,
                    "status": r.status,
                    "duration": round(r.duration, 4),
                    "message": r.message,
                    "category": r.category,
                    "file": r.file_path,
                }
                for r in self.results
            ],
            "coverage_gaps": self.coverage_gaps,
            "recommendations": self.recommendations,
        }

    def save(self, output_path: Path) -> None:
        """Save report to JSON file."""
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class TestOrchestrator:
    """Orchestrate comprehensive test execution."""

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self.project_root = project_root or Path(".")
        self.report = TestSuiteReport(
            timestamp=datetime.now().isoformat(),
        )

    def run_test_category(
        self, category: str, markers: Optional[List[str]] = None
    ) -> subprocess.CompletedProcess:
        """Run tests for a specific category."""
        cmd = ["python3", "-m", "pytest", "tests/", "-v"]

        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])

        cmd.extend(
            [
                "--tb=short",
            ]
        )

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.project_root,
        )

        return result

    def run_all_tests(self) -> TestSuiteReport:
        """Run complete test suite."""
        print("=" * 70)
        print("APGI COMPREHENSIVE TEST SUITE")
        print("=" * 70)

        # Run unit tests
        print("\n[1/5] Running Adversarial Unit Tests...")
        self._run_adversarial_tests()

        # Run integration tests
        print("\n[2/5] Running Integration Tests...")
        self._run_integration_tests()

        # Run security tests
        print("\n[3/5] Running Security Tests...")
        self._run_security_tests()

        # Run performance tests
        print("\n[4/5] Running Performance Tests...")
        self._run_performance_tests()

        # Run E2E tests
        print("\n[5/5] Running End-to-End Tests...")
        self._run_e2e_tests()

        # Generate coverage analysis
        print("\n[*] Analyzing Coverage...")
        self._analyze_coverage()

        # Generate recommendations
        self._generate_recommendations()

        return self.report

    def _run_adversarial_tests(self) -> None:
        """Run adversarial unit tests."""
        result = self.run_test_category(
            "adversarial",
            markers=["adversarial or boundary"],
        )
        self._parse_pytest_results(result, "adversarial")

    def _run_integration_tests(self) -> None:
        """Run integration tests."""
        result = self.run_test_category(
            "integration",
            markers=["integration"],
        )
        self._parse_pytest_results(result, "integration")

    def _run_security_tests(self) -> None:
        """Run security tests."""
        result = self.run_test_category(
            "security",
            markers=["security"],
        )
        self._parse_pytest_results(result, "security")

    def _run_performance_tests(self) -> None:
        """Run performance tests."""
        result = self.run_test_category(
            "performance",
            markers=["performance"],
        )
        self._parse_pytest_results(result, "performance")

    def _run_e2e_tests(self) -> None:
        """Run end-to-end tests."""
        result = self.run_test_category(
            "e2e",
            markers=["e2e"],
        )
        self._parse_pytest_results(result, "e2e")

    def _parse_pytest_results(
        self, result: subprocess.CompletedProcess, category: str
    ) -> None:
        """Parse pytest output and update report."""
        # Parse pytest output directly
        # Look for lines like: tests/test_file.py::TestClass::test_method PASSED [ 50%]
        # Or: tests/test_file.py::TestClass::test_method[param] PASSED [ 50%]
        test_pattern = re.compile(
            r"^(?P<file>[\w\-/]+\.py)::(?P<test>[^:]+)::(?P<name>[\w\[\]_\.\-]+)\s+(?P<status>PASSED|FAILED|SKIPPED|ERROR)"
        )

        for line in result.stdout.split("\n"):
            match = test_pattern.match(line.strip())
            if match:
                file_path = match.group("file")
                test_name = f"{match.group('test')}::{match.group('name')}"
                status = match.group("status").lower()

                test_result = TestResult(
                    name=test_name,
                    status=status,
                    duration=0.0,  # Duration not available in short output
                    message="",
                    category=category,
                    file_path=file_path,
                )
                self.report.results.append(test_result)

                # Update counts
                self.report.total_tests += 1
                if status == "passed":
                    self.report.passed += 1
                elif status == "failed":
                    self.report.failed += 1
                elif status == "skipped":
                    self.report.skipped += 1
                else:
                    self.report.errors += 1

    def _analyze_coverage(self) -> None:
        """Analyze coverage data."""
        # Check for coverage report
        coverage_path = self.project_root / "coverage.json"
        if coverage_path.exists():
            from tests.coverage_config import CoverageAnalyzer

            analyzer = CoverageAnalyzer()
            analyzer.load_from_json(coverage_path)
            gaps = analyzer.identify_gaps()

            # Update report
            for gap in gaps:
                gap_dict = {
                    "file": gap.file_path,
                    "lines": f"{gap.line_start}-{gap.line_end}",
                    "reason": gap.reason,
                    "suggested_test": gap.suggested_test,
                }
                gap_key = f"gap_{gap.line_start}_{gap.line_end}"
                self.report.coverage_gaps[gap_key] = gap_dict

        # Check if analyzer.coverage_data exists and has totals
        if hasattr(analyzer, "coverage_data") and isinstance(
            analyzer.coverage_data, dict
        ):
            totals = analyzer.coverage_data.get("totals")
            if totals and isinstance(totals, dict):
                self.report.coverage_percent = totals.get("percent_covered", 0.0)
                self.report.branch_coverage = totals.get(
                    "percent_covered_branches", 0.0
                )

    def _generate_recommendations(self) -> None:
        """Generate test recommendations."""
        recommendations = []

        # Coverage-based recommendations
        if self.report.coverage_percent < 90:
            recommendations.append(
                f"Increase line coverage from {self.report.coverage_percent:.1f}% to 90%+"
            )

        if self.report.branch_coverage < 80:
            recommendations.append(
                f"Increase branch coverage from {self.report.branch_coverage:.1f}% to 80%+"
            )

        # Test failure recommendations
        if self.report.failed > 0:
            recommendations.append(
                f"Fix {self.report.failed} failing tests before merge"
            )

        # Coverage gap recommendations
        if len(self.report.coverage_gaps) > 20:
            recommendations.append(
                f"Address {len(self.report.coverage_gaps)} coverage gaps, prioritizing core modules"
            )

        # Add specific recommendations based on gaps
        error_gaps = sum(
            1
            for g in self.report.coverage_gaps
            if isinstance(g, dict) and "error" in str(g.get("reason", "")).lower()
        )
        if error_gaps > 5:
            recommendations.append(
                f"Add {error_gaps} tests for exception handling paths"
            )

        if not recommendations:
            recommendations.append(
                "Test suite is comprehensive - maintain current coverage levels"
            )

        self.report.recommendations = recommendations

    def print_summary(self) -> None:
        """Print test execution summary."""
        print("\n" + "=" * 70)
        print("TEST EXECUTION SUMMARY")
        print("=" * 70)

        print(f"\nTotal Tests: {self.report.total_tests}")
        print(f"  Passed: {self.report.passed} ({self._pct(self.report.passed)}%)")
        print(f"  Failed: {self.report.failed} ({self._pct(self.report.failed)}%)")
        print(f"  Skipped: {self.report.skipped} ({self._pct(self.report.skipped)}%)")
        print(f"  Errors: {self.report.errors} ({self._pct(self.report.errors)}%)")

        print("\nCoverage:")
        print(f"  Line Coverage: {self.report.coverage_percent:.2f}%")
        print(f"  Branch Coverage: {self.report.branch_coverage:.2f}%")
        print(f"  Coverage Gaps: {len(self.report.coverage_gaps)}")

        print(f"\nDuration: {self.report.duration:.2f} seconds")

        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        for i, rec in enumerate(self.report.recommendations, 1):
            print(f"{i}. {rec}")

        print("\n" + "=" * 70)

    def _pct(self, count: int) -> float:
        """Calculate percentage."""
        if self.report.total_tests == 0:
            return 0.0
        return round(count / self.report.total_tests * 100, 1)

    def save_reports(self, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Save all reports to output directory."""
        if output_dir is None:
            output_dir = self.project_root / "test_reports"

        output_dir.mkdir(exist_ok=True)

        # Save main report
        report_path = output_dir / "test_report.json"
        self.report.save(report_path)

        # Save coverage gaps separately
        gaps_path = output_dir / "coverage_gaps.json"
        if self.report.coverage_gaps:
            with open(gaps_path, "w") as f:
                json.dump(self.report.coverage_gaps, f, indent=2)

        return {
            "main_report": report_path,
            "coverage_gaps": gaps_path,
        }


def main() -> int:
    """Main entry point for test orchestration."""
    orchestrator = TestOrchestrator()

    # Run all tests
    orchestrator.run_all_tests()

    # Print summary
    orchestrator.print_summary()

    # Save reports
    report_paths = orchestrator.save_reports()

    print("\nReports saved to:")
    for name, path in report_paths.items():
        if path:
            print(f"  - {name}: {path}")

    # Return exit code based on results
    if orchestrator.report.failed > 0 or orchestrator.report.errors > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
