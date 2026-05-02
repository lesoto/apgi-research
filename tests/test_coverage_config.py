"""
Tests for coverage_config.py - coverage configuration module.
"""

import json

from tests.coverage_config import (
    CoverageAnalyzer,
    CoverageConfig,
    CoverageGap,
    analyze_coverage_report,
    generate_coverage_config,
)


class TestCoverageConfig:
    """Tests for CoverageConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CoverageConfig()
        assert config.source_dirs == ["."]
        assert "*.py" in config.include_patterns
        assert config.line_coverage_threshold == 90.0
        assert config.branch_coverage_threshold == 80.0
        assert "html" in config.report_formats

    def test_to_coverage_rc(self):
        """Test generation of coverage.rc content."""
        config = CoverageConfig()
        rc_content = config.to_coverage_rc()
        assert "[run]" in rc_content
        assert "source = " in rc_content
        assert "branch = True" in rc_content
        assert "[report]" in rc_content
        assert "fail_under = 90.0" in rc_content
        assert "[html]" in rc_content
        assert "directory = htmlcov" in rc_content

    def test_to_pytest_config(self):
        """Test generation of pytest config args."""
        config = CoverageConfig()
        args = config.to_pytest_config()
        assert "--cov=." in args
        assert "--cov-fail-under=90.0" in args
        assert "--cov-report=html" in args


class TestCoverageGap:
    """Tests for CoverageGap dataclass."""

    def test_coverage_gap_creation(self):
        """Test creating a CoverageGap instance."""
        gap = CoverageGap(
            file_path="test.py",
            line_start=10,
            line_end=20,
            line_count=11,
            reason="test reason",
            suggested_test="add test",
        )
        assert gap.file_path == "test.py"
        assert gap.line_start == 10
        assert gap.line_end == 20
        assert gap.line_count == 11
        assert gap.reason == "test reason"
        assert gap.suggested_test == "add test"


class TestCoverageAnalyzer:
    """Tests for CoverageAnalyzer class."""

    def test_init_empty(self):
        """Test initialization with empty data."""
        analyzer = CoverageAnalyzer()
        assert analyzer.coverage_data == {}
        assert analyzer.gaps == []

    def test_init_with_data(self):
        """Test initialization with coverage data."""
        data = {"files": {"test.py": {"missing_lines": [1, 2, 3]}}}
        analyzer = CoverageAnalyzer(data)
        assert analyzer.coverage_data == data

    def test_load_from_json(self, tmp_path):
        """Test loading coverage data from JSON file."""
        json_file = tmp_path / "coverage.json"
        test_data = {"files": {"test.py": {"missing_lines": [1, 2]}}}
        json_file.write_text(json.dumps(test_data))

        analyzer = CoverageAnalyzer()
        analyzer.load_from_json(json_file)
        assert analyzer.coverage_data == test_data

    def test_identify_gaps(self):
        """Test identifying coverage gaps."""
        data = {
            "files": {
                "test.py": {"missing_lines": [1, 2, 3, 10, 11]},
                "test2.py": {"missing_lines": [5]},
            }
        }
        analyzer = CoverageAnalyzer(data)
        gaps = analyzer.identify_gaps()

        assert (
            len(gaps) == 3
        )  # Three groups: test.py (1-3), test.py (10-11), test2.py (5-5)
        assert gaps[0].file_path == "test.py"
        assert gaps[0].line_start == 1
        assert gaps[0].line_end == 3

    def test_group_consecutive_lines(self):
        """Test grouping consecutive line numbers."""
        analyzer = CoverageAnalyzer()
        lines = [1, 2, 3, 5, 6, 10]
        groups = analyzer._group_consecutive_lines(lines)
        assert groups == [(1, 3), (5, 6), (10, 10)]

    def test_group_consecutive_lines_empty(self):
        """Test grouping with empty list."""
        analyzer = CoverageAnalyzer()
        groups = analyzer._group_consecutive_lines([])
        assert groups == []

    def test_classify_gap_test_file(self):
        """Test gap classification for test files."""
        analyzer = CoverageAnalyzer()
        result = analyzer._classify_gap("test_something.py", 10, 20)
        assert "Test file" in result

    def test_classify_gap_init(self):
        """Test gap classification for __init__ files."""
        analyzer = CoverageAnalyzer()
        result = analyzer._classify_gap("module/__init__.py", 10, 20)
        assert "Module initialization" in result

    def test_classify_gap_exception(self):
        """Test classifying gap in exception handling code."""
        analyzer = CoverageAnalyzer()
        result = analyzer._classify_gap("module_exceptions.py", 10, 20)
        assert "Exception" in result or "error" in result.lower()

    def test_classify_gap_module_level(self):
        """Test gap classification for early lines."""
        analyzer = CoverageAnalyzer()
        result = analyzer._classify_gap("module.py", 5, 15)
        assert "Module-level" in result

    def test_classify_gap_general(self):
        """Test gap classification for general cases."""
        analyzer = CoverageAnalyzer()
        result = analyzer._classify_gap("module.py", 50, 60)
        assert "Logic branches" in result

    def test_suggest_test_exception(self):
        """Test test suggestion for exception files."""
        analyzer = CoverageAnalyzer()
        result = analyzer._suggest_test("test_exceptions.py", 10, 20)
        assert "exception handling" in result

    def test_suggest_test_validation(self):
        """Test test suggestion for validation files."""
        analyzer = CoverageAnalyzer()
        result = analyzer._suggest_test("test_validation.py", 10, 20)
        assert "boundary" in result

    def test_suggest_test_utils(self):
        """Test test suggestion for utils files."""
        analyzer = CoverageAnalyzer()
        result = analyzer._suggest_test("test_utils.py", 10, 20)
        assert "unit test" in result

    def test_generate_report(self):
        """Test generating comprehensive coverage report."""
        data = {
            "totals": {"covered_lines": 100},
            "files": {"test.py": {"missing_lines": [1, 2, 3]}},
        }
        analyzer = CoverageAnalyzer(data)
        analyzer.identify_gaps()
        report = analyzer.generate_report()

        assert "summary" in report
        assert "coverage_percent" in report["summary"]
        assert "gaps_by_file" in report
        assert "recommendations" in report

    def test_group_gaps_by_file(self):
        """Test grouping gaps by file path."""
        data = {"files": {"test.py": {"missing_lines": [1, 2]}}}
        analyzer = CoverageAnalyzer(data)
        analyzer.identify_gaps()
        by_file = analyzer._group_gaps_by_file()

        assert "test.py" in by_file
        assert len(by_file["test.py"]) == 1

    def test_generate_recommendations_error_handling(self):
        """Test recommendations for error handling gaps."""
        analyzer = CoverageAnalyzer()
        # Create non-consecutive lines to generate multiple gaps (>5 needed for recommendation)
        missing = [1, 2, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51]  # 6 separate gaps
        analyzer.coverage_data = {
            "files": {"module_exceptions.py": {"missing_lines": missing}}
        }
        analyzer.identify_gaps()  # Populate gaps list
        recs = analyzer._generate_recommendations()
        assert any("exception handling" in r.lower() for r in recs)

    def test_generate_recommendations_boundary(self):
        """Test recommendations for boundary gaps."""
        analyzer = CoverageAnalyzer()
        analyzer.gaps = [
            CoverageGap("test.py", 10, 20, 11, "boundary values", "add test")
        ]
        recs = analyzer._generate_recommendations()
        assert any("boundary" in r for r in recs)

    def test_generate_recommendations_general(self):
        """Test general recommendations."""
        analyzer = CoverageAnalyzer()
        analyzer.gaps = [
            CoverageGap("test.py", 10, 20, 11, "logic", "add test") for _ in range(25)
        ]
        recs = analyzer._generate_recommendations()
        assert any("prioritizing" in r for r in recs)

    def test_generate_recommendations_good_coverage(self):
        """Test recommendations when coverage is good."""
        analyzer = CoverageAnalyzer()
        analyzer.gaps = []
        recs = analyzer._generate_recommendations()
        assert any("good" in r.lower() for r in recs)


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_generate_coverage_config_default_path(self, tmp_path):
        """Test generating coverage config with default path."""
        import os

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            path = generate_coverage_config()
            assert path.exists()
            assert path.name == ".coveragerc"
            content = path.read_text()
            assert "[run]" in content
        finally:
            os.chdir(original_dir)

    def test_generate_coverage_config_custom_path(self, tmp_path):
        """Test generating coverage config with custom path."""
        custom_path = tmp_path / "custom_coverage.rc"
        result = generate_coverage_config(custom_path)
        assert result == custom_path
        assert custom_path.exists()

    def test_analyze_coverage_report(self, tmp_path):
        """Test analyzing a coverage report."""
        json_file = tmp_path / "coverage.json"
        test_data = {
            "totals": {"covered_lines": 100},
            "files": {"test.py": {"missing_lines": [1]}},
        }
        json_file.write_text(json.dumps(test_data))

        report = analyze_coverage_report(json_file)
        assert "summary" in report
        assert "gaps_by_file" in report
