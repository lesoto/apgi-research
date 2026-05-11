"""
================================================================================
COVERAGE CONFIGURATION
================================================================================

This module provides comprehensive coverage configuration for:
- Line coverage tracking
- Branch coverage analysis
- Path coverage measurement
- Missing coverage reporting
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CoverageConfig:
    """Configuration for comprehensive test coverage."""

    # Directories to measure
    source_dirs: List[str] = field(default_factory=lambda: ["."])

    # Files to include
    include_patterns: List[str] = field(
        default_factory=lambda: [
            "*.py",
        ]
    )

    # Files to exclude
    exclude_patterns: List[str] = field(
        default_factory=lambda: [
            "*/tests/*",
            "*/test_*",
            "*/__pycache__/*",
            "*/venv/*",
            "*/.venv/*",
            "*/node_modules/*",
            "setup.py",
            "conftest.py",
        ]
    )

    # Coverage thresholds
    line_coverage_threshold: float = 90.0
    branch_coverage_threshold: float = 80.0

    # Report formats
    report_formats: List[str] = field(
        default_factory=lambda: ["html", "xml", "json", "term-missing"]
    )

    def to_coverage_rc(self) -> str:
        """Generate coverage.rc configuration content."""
        lines = [
            "[run]",
            "source = " + ", ".join(self.source_dirs),
            "branch = True",
            "parallel = True",
            "concurrency = thread, multiprocessing",
            "omit = " + "\n    ".join(self.exclude_patterns),
            "",
            "[report]",
            f"fail_under = {self.line_coverage_threshold}",
            "skip_covered = False",
            "skip_empty = True",
            "show_missing = True",
            "exclude_lines =",
            "    pragma: no cover",
            "    def __repr__",
            "    raise NotImplementedError",
            "    if __name__ == .__main__.:",
            "    if TYPE_CHECKING:",
            "    @abstract",
            "    @abc.abstractmethod",
            "",
            "[html]",
            "directory = htmlcov",
            "",
            "[xml]",
            "output = coverage.xml",
            "",
            "[json]",
            "output = coverage.json",
        ]
        return "\n".join(lines)

    def to_pytest_config(self) -> List[str]:
        """Generate pytest addopts for coverage."""
        args = [
            "--cov=.",
            f"--cov-fail-under={self.line_coverage_threshold}",
            "--cov-report=html",
            "--cov-report=xml",
            "--cov-report=term-missing",
        ]
        for pattern in self.exclude_patterns:
            args.append(f"--cov-omit={pattern}")
        return args


@dataclass
class CoverageGap:
    """Represents a gap in code coverage."""

    file_path: str
    line_start: int
    line_end: int
    line_count: int
    reason: str = ""
    suggested_test: str = ""


class CoverageAnalyzer:
    """Analyze coverage reports and identify gaps."""

    def __init__(self, coverage_data: Optional[Dict[str, Any]] = None) -> None:
        self.coverage_data = coverage_data or {}
        self.gaps: List[CoverageGap] = []

    def load_from_json(self, json_path: Path) -> None:
        """Load coverage data from JSON report."""
        with open(json_path) as f:
            self.coverage_data = json.load(f)

    def identify_gaps(self) -> List[CoverageGap]:
        """Identify coverage gaps from coverage data."""
        self.gaps = []

        files = self.coverage_data.get("files", {})
        for file_path, file_data in files.items():
            missing_lines = file_data.get("missing_lines", [])
            if not missing_lines:
                continue

            # Group consecutive missing lines
            groups = self._group_consecutive_lines(missing_lines)

            for start, end in groups:
                gap = CoverageGap(
                    file_path=file_path,
                    line_start=start,
                    line_end=end,
                    line_count=end - start + 1,
                    reason=self._classify_gap(file_path, start, end),
                    suggested_test=self._suggest_test(file_path, start, end),
                )
                self.gaps.append(gap)

        return self.gaps

    def _group_consecutive_lines(self, lines: List[int]) -> List[Tuple[int, int]]:
        """Group consecutive line numbers into ranges."""
        if not lines:
            return []

        groups = []
        start = lines[0]
        prev = lines[0]

        for line in lines[1:]:
            if line != prev + 1:
                groups.append((start, prev))
                start = line
            prev = line

        groups.append((start, prev))
        return groups

    def _classify_gap(self, file_path: str, line_start: int, line_end: int) -> str:
        """Classify the type of coverage gap."""
        # Check file type
        if "test" in file_path.lower():
            return "Test file not covered (acceptable)"
        elif "__init__" in file_path:
            return "Module initialization"
        elif "exception" in file_path.lower() or "error" in file_path.lower():
            return "Exception/error handling paths"
        elif line_start < 20:
            return "Module-level code or imports"
        else:
            return "Logic branches or edge cases"

    def _suggest_test(self, file_path: str, line_start: int, line_end: int) -> str:
        """Suggest what type of test would cover this gap."""
        if "exception" in file_path.lower():
            return "Add test for exception handling with invalid inputs"
        elif "validation" in file_path.lower():
            return "Add test with boundary/invalid values"
        elif "utils" in file_path.lower():
            return "Add unit test for utility function"
        else:
            return f"Add test covering lines {line_start}-{line_end}"

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive coverage report."""
        total_lines = self.coverage_data.get("totals", {}).get(
            "covered_lines", 0
        ) + sum(g.line_count for g in self.gaps)

        coverage_pct = (
            (total_lines - sum(g.line_count for g in self.gaps)) / total_lines * 100
            if total_lines > 0
            else 0
        )

        return {
            "summary": {
                "coverage_percent": round(coverage_pct, 2),
                "total_gaps": len(self.gaps),
                "total_missing_lines": sum(g.line_count for g in self.gaps),
            },
            "gaps_by_file": self._group_gaps_by_file(),
            "recommendations": self._generate_recommendations(),
        }

    def _group_gaps_by_file(self) -> Dict[str, List[Dict[str, Any]]]:
        """Group gaps by file path."""
        by_file: Dict[str, List[Dict[str, Any]]] = {}
        for gap in self.gaps:
            if gap.file_path not in by_file:
                by_file[gap.file_path] = []
            by_file[gap.file_path].append(
                {
                    "lines": f"{gap.line_start}-{gap.line_end}",
                    "count": gap.line_count,
                    "reason": gap.reason,
                    "suggested_test": gap.suggested_test,
                }
            )
        return by_file

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for improving coverage."""
        recommendations = []

        # Check for common patterns
        error_handling_gaps = sum(
            1
            for g in self.gaps
            if "exception" in g.file_path.lower() or "error" in g.reason.lower()
        )
        if error_handling_gaps > 5:
            recommendations.append(
                f"Add tests for {error_handling_gaps} exception handling paths"
            )

        boundary_gaps = sum(1 for g in self.gaps if "boundary" in g.reason.lower())
        if boundary_gaps > 0:
            recommendations.append(f"Add {boundary_gaps} boundary value tests")

        # General recommendations
        if len(self.gaps) > 20:
            recommendations.append(
                "Consider prioritizing coverage gaps in core modules first"
            )

        if not recommendations:
            recommendations.append("Coverage is good - consider adding edge case tests")

        return recommendations


def generate_coverage_config(output_path: Optional[Path] = None) -> Path:
    """Generate and save coverage configuration."""
    config = CoverageConfig()

    if output_path is None:
        output_path = Path(".coveragerc")

    output_path.write_text(config.to_coverage_rc())
    return output_path


def analyze_coverage_report(coverage_json_path: Path) -> Dict[str, Any]:
    """Analyze a coverage report and return structured analysis."""
    analyzer = CoverageAnalyzer()
    analyzer.load_from_json(coverage_json_path)
    analyzer.identify_gaps()
    return analyzer.generate_report()


if __name__ == "__main__":
    # Generate coverage configuration
    config_path = generate_coverage_config()
    print(f"Generated coverage config: {config_path}")
