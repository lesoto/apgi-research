"""
Comprehensive tests for analyze_experiments module.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from analyze_experiments import (
    EXPERIMENT_RESULTS,
    analyze_apgi_metrics,
    generate_fixes,
    generate_html_report,
    get_apgi_experiments,
    identify_issues,
    main,
)


class TestExperimentData:
    """Tests for experiment data structure."""

    def test_experiment_results_structure(self):
        """Test that EXPERIMENT_RESULTS has expected structure."""
        assert isinstance(EXPERIMENT_RESULTS, dict)
        assert len(EXPERIMENT_RESULTS) > 0

        # Check that key experiments exist
        assert "Artificial Grammar Learning" in EXPERIMENT_RESULTS
        assert "Attentional Blink" in EXPERIMENT_RESULTS
        assert "Ai Benchmarking" in EXPERIMENT_RESULTS

    def test_experiment_data_completeness(self):
        """Test that experiment data has required fields."""
        for exp_name, data in EXPERIMENT_RESULTS.items():
            if data.get("status") == "Success":
                # Should have primary_metric for successful experiments
                assert "primary_metric" in data
                assert isinstance(data["primary_metric"], (int, float))


class TestGetAPGIExperiments:
    """Tests for get_apgi_experiments function."""

    def test_get_apgi_experiments(self):
        """Test getting APGI experiments."""
        apgi_exps = get_apgi_experiments()

        assert isinstance(apgi_exps, dict)
        assert len(apgi_exps) > 0

        # Check that returned experiments have APGI metrics
        for exp_name, data in apgi_exps.items():
            assert "ignition_rate" in data
            assert "mean_surprise" in data
            assert "metabolic_cost" in data

    def test_get_apgi_experiments_filtering(self):
        """Test that only experiments with APGI metrics are returned."""
        apgi_exps = get_apgi_experiments()

        # Should not include experiments without APGI metrics
        for exp_name, data in apgi_exps.items():
            assert "ignition_rate" in data, f"{exp_name} missing ignition_rate"

    def test_get_apgi_experiments_data_types(self):
        """Test that returned data has correct types."""
        apgi_exps = get_apgi_experiments()

        for exp_name, data in apgi_exps.items():
            if "ignition_rate" in data:
                assert isinstance(data["ignition_rate"], (int, float))
            if "mean_surprise" in data:
                assert isinstance(data["mean_surprise"], (int, float))
            if "metabolic_cost" in data:
                assert isinstance(data["metabolic_cost"], (int, float))


class TestAnalyzeAPGIMetrics:
    """Tests for analyze_apgi_metrics function."""

    def test_analyze_apgi_metrics(self):
        """Test APGI metrics analysis."""
        analysis = analyze_apgi_metrics()

        assert isinstance(analysis, dict)
        assert "summary" in analysis
        assert "detailed_analysis" in analysis
        assert "correlations" in analysis

    def test_analyze_apgi_metrics_summary(self):
        """Test analysis summary structure."""
        analysis = analyze_apgi_metrics()

        assert "total_experiments" in analysis
        assert "summary" in analysis
        assert "avg_ignition_rate" in analysis["summary"]
        assert "avg_metabolic_cost" in analysis["summary"]
        assert "avg_surprise" in analysis["summary"]

    def test_analyze_apgi_metrics_detailed(self):
        """Test detailed analysis structure."""
        analysis = analyze_apgi_metrics()
        detailed = analysis["detailed_analysis"]

        assert isinstance(detailed, dict)
        # Should have analysis for each APGI experiment
        assert len(detailed) > 0

    def test_analyze_apgi_metrics_correlations(self):
        """Test correlation analysis."""
        analysis = analyze_apgi_metrics()
        correlations = analysis["correlations"]

        assert isinstance(correlations, dict)
        # Should have correlation coefficients
        for corr_name, corr_value in correlations.items():
            assert isinstance(corr_value, (int, float))
            assert -1 <= corr_value <= 1

    def test_analyze_apgi_metrics_values(self):
        """Test that analysis values are reasonable."""
        analysis = analyze_apgi_metrics()
        summary = analysis["summary"]

        # Check that averages are reasonable
        assert 0 <= summary["avg_ignition_rate"] <= 100
        assert summary["avg_metabolic_cost"] >= 0
        assert summary["avg_surprise"] >= 0


class TestIdentifyIssues:
    """Tests for identify_issues function."""

    def test_identify_issues(self):
        """Test issue identification."""
        issues = identify_issues()

        assert isinstance(issues, list)
        # Each issue should be a dictionary
        for issue in issues:
            assert isinstance(issue, dict)
            assert "experiment" in issue
            assert "issue_type" in issue
            assert "description" in issue

    def test_identify_issues_structure(self):
        """Test issue structure."""
        issues = identify_issues()

        for issue in issues:
            assert "experiment" in issue
            assert "issue_type" in issue
            assert "description" in issue
            assert "severity" in issue
            assert "suggested_fix" in issue

    def test_identify_issues_severity_levels(self):
        """Test that severity levels are valid."""
        issues = identify_issues()

        valid_severities = ["low", "medium", "high", "critical"]
        for issue in issues:
            assert issue["severity"] in valid_severities

    def test_identify_issues_types(self):
        """Test that issue types are valid."""
        issues = identify_issues()

        # Should identify various types of issues
        issue_types = set(issue["issue_type"] for issue in issues)
        assert len(issue_types) > 0


class TestGenerateFixes:
    """Tests for generate_fixes function."""

    def test_generate_fixes(self):
        """Test fix generation."""
        fixes = generate_fixes()

        assert isinstance(fixes, dict)
        # Each fix should be a list of strings
        for exp_name, fix_list in fixes.items():
            assert isinstance(fix_list, list)
            for fix in fix_list:
                assert isinstance(fix, str)

    def test_generate_fixes_coverage(self):
        """Test that fixes are generated for experiments with issues."""
        issues = identify_issues()
        fixes = generate_fixes()

        # Should have fixes for experiments with issues
        experiments_with_issues = set(issue["experiment"] for issue in issues)
        experiments_with_fixes = set(fixes.keys())

        # All experiments with issues should have fixes
        assert experiments_with_issues.issubset(experiments_with_fixes)

    def test_generate_fixes_content(self):
        """Test that fix content is meaningful."""
        fixes = generate_fixes()
        for exp_name, fix_list in fixes.items():
            assert len(fix_list) > 0  # Should be descriptive
            assert not fix_list[0].startswith(" ")  # Should not start with whitespace


class TestGenerateHTMLReport:
    """Tests for generate_html_report function."""

    def test_generate_html_report(self):
        """Test HTML report structure."""
        # Use real function instead of mock
        analysis = analyze_apgi_metrics()
        issues = identify_issues()
        fixes = generate_fixes()
        html = generate_html_report(analysis, issues, fixes)

        assert isinstance(html, str)
        assert html.startswith("<!DOCTYPE html>")
        assert html.strip().endswith("</html>")

    def test_generate_html_report_structure(self):
        """Test HTML report structure."""
        analysis: Dict[str, Any] = {"total_experiments": 10, "metrics_summary": {}}
        issues: List[Dict[str, Any]] = [
            {
                "experiment": "test",
                "issue_type": "test",
                "severity": "medium",
                "issues": ["issue1"],
            }
        ]
        fixes: Dict[str, List[str]] = {"test": ["fix1", "fix2"]}

        html = generate_html_report(analysis, issues, fixes)

        # Should contain key sections
        assert "<title>" in html
        assert "<body>" in html
        assert "</body>" in html
        assert "</html>" in html

    def test_generate_html_report_content(self):
        """Test HTML report contains expected content."""
        analysis: Dict[str, Any] = {"total_experiments": 10, "metrics_summary": {}}
        issues: List[Dict[str, Any]] = [
            {
                "experiment": "test",
                "issue_type": "test",
                "severity": "medium",
                "issues": ["issue1"],
            }
        ]
        fixes: Dict[str, List[str]] = {"test": ["fix1", "fix2"]}

        html = generate_html_report(analysis, issues, fixes)

        # Should contain analysis data
        assert "10" in html  # total_experiments value
        assert "test" in html  # experiment name

    def test_generate_html_report_css(self):
        """Test HTML report contains CSS styling."""
        analysis: Dict[str, Any] = {"total_experiments": 10, "metrics_summary": {}}
        issues: List[Dict[str, Any]] = [
            {
                "experiment": "test",
                "issue_type": "test",
                "severity": "medium",
                "issues": ["issue1"],
            }
        ]
        fixes: Dict[str, List[str]] = {"test": ["fix1", "fix2"]}

        html = generate_html_report(analysis, issues, fixes)

        # Should contain CSS
        assert "<style>" in html or "style=" in html

    def test_generate_html_report_empty_data(self):
        """Test HTML report with empty data."""
        analysis: Dict[str, Any] = {"total_experiments": 0, "metrics_summary": {}}
        issues: List[Dict[str, Any]] = []
        fixes: Dict[str, List[str]] = {}

        html = generate_html_report(analysis, issues, fixes)

        assert isinstance(html, str)
        assert len(html) > 0


class TestMainFunction:
    """Tests for main function."""

    @patch("builtins.print")
    @patch("analyze_experiments.generate_html_report")
    @patch("analyze_experiments.analyze_apgi_metrics")
    @patch("analyze_experiments.identify_issues")
    @patch("analyze_experiments.generate_fixes")
    def test_main_function(
        self, mock_fixes, mock_issues, mock_analysis, mock_html, mock_print
    ):
        """Test main function execution."""
        # Setup mocks
        mock_analysis.return_value = {
            "total_experiments": 10,
            "metrics_summary": {},
        }
        mock_issues.return_value = []
        mock_fixes.return_value = {}
        mock_html.return_value = "<html></html>"

        # Call main
        main()

        # Verify function calls
        mock_analysis.assert_called_once()
        mock_issues.assert_called_once()
        mock_fixes.assert_called_once()
        mock_html.assert_called_once()

    @patch("builtins.print")
    @patch("analyze_experiments.generate_html_report")
    @patch("analyze_experiments.analyze_apgi_metrics")
    @patch("analyze_experiments.identify_issues")
    @patch("analyze_experiments.generate_fixes")
    def test_main_file_output(
        self, mock_fixes, mock_issues, mock_analysis, mock_html, mock_print
    ):
        """Test main function writes to file."""
        # Setup mocks
        mock_analysis.return_value = {"total_experiments": 10, "metrics_summary": {}}
        mock_issues.return_value = []
        mock_fixes.return_value = {}
        mock_html.return_value = "<html></html>"

        # Mock open
        with patch("builtins.open", MagicMock()) as mock_open:
            # Call main
            main()

            # Verify file write
            mock_open.assert_called_once()
            args, kwargs = mock_open.call_args
            assert str(args[0]) == "apgi_analysis_report.html"
            assert args[1] == "w"


class TestModuleIntegration:
    """Integration tests for the module."""

    def test_full_analysis_workflow(self):
        """Test complete analysis workflow."""
        # Get APGI experiments
        apgi_exps = get_apgi_experiments()

        # Analyze metrics
        analysis = analyze_apgi_metrics()

        # Identify issues
        issues = identify_issues()

        # Generate fixes
        fixes = generate_fixes()

        # Generate HTML report
        html = generate_html_report(analysis, issues, fixes)

        # Verify workflow
        assert isinstance(apgi_exps, dict)
        assert isinstance(analysis, dict)
        assert isinstance(issues, list)
        assert isinstance(fixes, dict)
        assert isinstance(html, str)

    def test_data_consistency(self):
        """Test data consistency across functions."""
        apgi_exps = get_apgi_experiments()
        analysis = analyze_apgi_metrics()

        # Number of experiments should be consistent
        assert len(apgi_exps) == analysis["summary"]["total_experiments"]

    def test_error_handling_missing_data(self):
        """Test handling of missing experiment data."""
        # This should not raise an exception even with incomplete data
        try:
            analysis = analyze_apgi_metrics()
            issues = identify_issues()
            fixes = generate_fixes()

            assert isinstance(analysis, dict)
            assert isinstance(issues, list)
            assert isinstance(fixes, dict)
        except Exception as e:
            pytest.fail(f"Analysis should handle missing data gracefully: {e}")


class TestModuleConstants:
    """Tests for module constants."""

    def test_experiment_results_constant(self):
        """Test EXPERIMENT_RESULTS constant."""
        assert isinstance(EXPERIMENT_RESULTS, dict)
        assert len(EXPERIMENT_RESULTS) > 0

        # Should have required experiments
        required_experiments = [
            "Artificial Grammar Learning",
            "Attentional Blink",
            "Ai Benchmarking",
        ]

        for exp in required_experiments:
            assert exp in EXPERIMENT_RESULTS, f"Missing required experiment: {exp}"

    def test_experiment_data_validation(self):
        """Test experiment data validation."""
        for exp_name, data in EXPERIMENT_RESULTS.items():
            assert isinstance(data, dict)
            assert "status" in data

            if data["status"] == "Success":
                # Successful experiments should have metrics
                assert "primary_metric" in data
                assert isinstance(data["primary_metric"], (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
