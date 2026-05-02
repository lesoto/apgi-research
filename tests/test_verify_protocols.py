"""
Tests for experiments/verify_protocols.py - experiment protocol verification script.
"""

from pathlib import Path
from unittest.mock import patch

from experiments.verify_protocols import (
    ADDITIONAL_EXPERIMENTS,
    EXPERIMENTS,
    VerificationResult,
    check_agent_editable_marker,
    check_import_structure,
    check_primary_metric,
    check_read_only_marker,
    check_time_budget,
    main,
    verify_experiment,
)


class TestCheckReadOnlyMarker:
    """Tests for check_read_only_marker function."""

    def test_full_marker_present(self, tmp_path):
        """Test when both READ-ONLY and Do not modify are present."""
        test_file = tmp_path / "prepare_test.py"
        test_file.write_text('"""\nREAD-ONLY file.\nDo not modify.\n"""\n')
        ok, msg = check_read_only_marker(test_file)
        assert ok is True
        assert "✅" in msg

    def test_partial_marker(self, tmp_path):
        """Test when READ-ONLY present but missing Do not modify."""
        test_file = tmp_path / "prepare_test.py"
        test_file.write_text('"""\nREAD-ONLY file.\n"""\n')
        ok, msg = check_read_only_marker(test_file)
        assert ok is True
        assert "⚠️" in msg

    def test_missing_marker(self, tmp_path):
        """Test when READ-ONLY marker is missing."""
        test_file = tmp_path / "prepare_test.py"
        test_file.write_text('"""\nSome other docstring.\n"""\n')
        ok, msg = check_read_only_marker(test_file)
        assert ok is False
        assert "❌" in msg

    def test_error_handling(self, tmp_path):
        """Test error handling when file can't be read."""
        test_file = tmp_path / "nonexistent.py"
        ok, msg = check_read_only_marker(test_file)
        assert ok is False
        assert "Error" in msg


class TestCheckAgentEditableMarker:
    """Tests for check_agent_editable_marker function."""

    def test_full_marker_present(self, tmp_path):
        """Test when AGENT-EDITABLE marker is present."""
        test_file = tmp_path / "run_test.py"
        test_file.write_text('"""\nAGENT-EDITABLE file.\nModify this file.\n"""\n')
        ok, msg = check_agent_editable_marker(test_file)
        assert ok is True
        assert "✅" in msg

    def test_alternative_marker(self, tmp_path):
        """Test when modifiable hint is present."""
        test_file = tmp_path / "run_test.py"
        test_file.write_text('"""\nThis file is modifiable.\n"""\n')
        ok, msg = check_agent_editable_marker(test_file)
        assert ok is True
        assert "⚠️" in msg

    def test_missing_marker(self, tmp_path):
        """Test when AGENT-EDITABLE marker is missing."""
        test_file = tmp_path / "run_test.py"
        test_file.write_text('"""\nSome other docstring.\n"""\n')
        ok, msg = check_agent_editable_marker(test_file)
        assert ok is False
        assert "❌" in msg

    def test_error_handling(self, tmp_path):
        """Test error handling when file can't be read."""
        test_file = tmp_path / "nonexistent.py"
        ok, msg = check_agent_editable_marker(test_file)
        assert ok is False
        assert "Error" in msg


class TestCheckTimeBudget:
    """Tests for check_time_budget function."""

    def test_correct_time_budget(self, tmp_path):
        """Test when TIME_BUDGET = 600 is correct."""
        test_file = tmp_path / "test.py"
        test_file.write_text("TIME_BUDGET = 600\n")
        ok, msg = check_time_budget(test_file, "prepare")
        assert ok is True
        assert "✅" in msg

    def test_incorrect_time_budget(self, tmp_path):
        """Test when TIME_BUDGET is wrong."""
        test_file = tmp_path / "test.py"
        test_file.write_text("TIME_BUDGET = 300\n")
        ok, msg = check_time_budget(test_file, "prepare")
        assert ok is False
        assert "❌" in msg
        assert "300" in msg

    def test_assertion_present(self, tmp_path):
        """Test when TIME_BUDGET assertion is present."""
        test_file = tmp_path / "test.py"
        test_file.write_text("assert TIME_BUDGET == 600\n")
        ok, msg = check_time_budget(test_file, "prepare")
        assert ok is True
        assert "✅" in msg

    def test_mentioned_in_comments(self, tmp_path):
        """Test when 600 is mentioned but not as constant."""
        test_file = tmp_path / "test.py"
        test_file.write_text("# Time budget: 600 seconds\n")
        ok, msg = check_time_budget(test_file, "prepare")
        assert ok is True
        assert "⚠️" in msg

    def test_missing_time_budget(self, tmp_path):
        """Test when TIME_BUDGET is missing."""
        test_file = tmp_path / "test.py"
        test_file.write_text("# No time budget here\n")
        ok, msg = check_time_budget(test_file, "prepare")
        assert ok is False
        assert "❌" in msg

    def test_error_handling(self, tmp_path):
        """Test error handling when file can't be read."""
        test_file = tmp_path / "nonexistent.py"
        ok, msg = check_time_budget(test_file, "prepare")
        assert ok is False
        assert "Error" in msg


class TestCheckPrimaryMetric:
    """Tests for check_primary_metric function."""

    def test_primary_metric_format(self, tmp_path):
        """Test when primary_metric: format is present."""
        test_file = tmp_path / "run_test.py"
        test_file.write_text('print("primary_metric: 0.5")\n')
        ok, msg = check_primary_metric(test_file, "stroop_effect")
        assert ok is True
        assert "✅" in msg

    def test_expected_metric_present(self, tmp_path):
        """Test when expected metric is present."""
        test_file = tmp_path / "run_test.py"
        test_file.write_text('print("interference_effect_ms: 50")\n')
        ok, msg = check_primary_metric(test_file, "stroop_effect")
        assert ok is True
        assert "interference_effect_ms" in msg

    def test_common_metric_present(self, tmp_path):
        """Test when common metric is present."""
        test_file = tmp_path / "run_test.py"
        test_file.write_text('print("accuracy: 0.95")\n')
        ok, msg = check_primary_metric(test_file, "unknown_experiment")
        assert ok is True
        assert "⚠️" in msg

    def test_missing_metric(self, tmp_path):
        """Test when no metric is present."""
        test_file = tmp_path / "run_test.py"
        test_file.write_text('print("hello")\n')
        ok, msg = check_primary_metric(test_file, "stroop_effect")
        assert ok is False
        assert "❌" in msg

    def test_error_handling(self, tmp_path):
        """Test error handling when file can't be read."""
        test_file = tmp_path / "nonexistent.py"
        ok, msg = check_primary_metric(test_file, "stroop_effect")
        assert ok is False
        assert "Error" in msg


class TestCheckImportStructure:
    """Tests for check_import_structure function."""

    def test_from_import(self, tmp_path):
        """Test when from prepare_module import is present."""
        test_file = tmp_path / "run_stroop_effect.py"
        test_file.write_text("from prepare_stroop_effect import TIME_BUDGET\n")
        ok, msg = check_import_structure(test_file, "stroop_effect")
        assert ok is True
        assert "✅" in msg
        assert "TIME_BUDGET" in msg

    def test_direct_import(self, tmp_path):
        """Test when direct import is present."""
        test_file = tmp_path / "run_stroop_effect.py"
        test_file.write_text("import prepare_stroop_effect\n")
        ok, msg = check_import_structure(test_file, "stroop_effect")
        assert ok is True
        assert "✅" in msg

    def test_missing_import(self, tmp_path):
        """Test when import is missing."""
        test_file = tmp_path / "run_stroop_effect.py"
        test_file.write_text("print('hello')\n")
        ok, msg = check_import_structure(test_file, "stroop_effect")
        assert ok is False
        assert "❌" in msg

    def test_igt_special_case(self, tmp_path):
        """Test IGT special case handling."""
        test_file = tmp_path / "run_igt.py"
        test_file.write_text("from prepare_igt import TIME_BUDGET\n")
        ok, msg = check_import_structure(test_file, "igt")
        assert ok is True
        assert "prepare_igt" in msg

    def test_error_handling(self, tmp_path):
        """Test error handling when file can't be read."""
        test_file = tmp_path / "nonexistent.py"
        ok, msg = check_import_structure(test_file, "stroop_effect")
        assert ok is False
        assert "Error" in msg


class TestVerifyExperiment:
    """Tests for verify_experiment function."""

    def test_missing_files(self, tmp_path):
        """Test when experiment files don't exist."""
        result = verify_experiment("nonexistent", tmp_path)
        assert len(result.failed) > 0
        assert "Missing" in result.failed[0]

    def test_existing_files_no_markers(self, tmp_path):
        """Test with existing files but missing markers."""
        prepare_file = tmp_path / "prepare_test.py"
        run_file = tmp_path / "run_test.py"
        prepare_file.write_text('"""Test."""\n')
        run_file.write_text('"""Test."""\n')

        result = verify_experiment("test", tmp_path)
        assert len(result.failed) > 0  # Should fail markers

    def test_complete_protocol(self, tmp_path):
        """Test with complete valid protocol."""
        prepare_file = tmp_path / "prepare_test.py"
        run_file = tmp_path / "run_test.py"

        prepare_file.write_text(
            '"""\nREAD-ONLY file.\nDo not modify.\n"""\nTIME_BUDGET = 600\n'
        )
        run_file.write_text(
            '"""\nAGENT-EDITABLE file.\nModify this file.\n"""\n'
            "TIME_BUDGET = 600\n"
            "from prepare_test import TIME_BUDGET\n"
            "print('primary_metric: 0.5')\n"
        )

        result = verify_experiment("test", tmp_path)
        assert len(result.passed) >= 5
        assert len(result.failed) == 0

    def test_igt_experiment(self, tmp_path):
        """Test IGT experiment special case."""
        prepare_file = tmp_path / "prepare_igt.py"
        run_file = tmp_path / "run_igt.py"

        prepare_file.write_text('"""READ-ONLY"""\nTIME_BUDGET = 600\n')
        run_file.write_text(
            '"""AGENT-EDITABLE"""\n'
            "TIME_BUDGET = 600\n"
            "from prepare_igt import TIME_BUDGET\n"
            "print('net_score: 10')\n"
        )

        result = verify_experiment("igt", tmp_path)
        assert len(result.passed) >= 5


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_creation(self):
        """Test creating a VerificationResult."""
        result = VerificationResult(
            experiment="test",
            passed=["1. Test passed"],
            failed=["2. Test failed"],
            warnings=["3. Warning"],
        )
        assert result.experiment == "test"
        assert len(result.passed) == 1
        assert len(result.failed) == 1
        assert len(result.warnings) == 1


class TestExperimentsList:
    """Tests for EXPERIMENTS and ADDITIONAL_EXPERIMENTS constants."""

    def test_experiments_list(self):
        """Test that EXPERIMENTS contains expected experiments."""
        assert "stroop_effect" in EXPERIMENTS
        assert "ai_benchmarking" in EXPERIMENTS
        assert "change_blindness" in EXPERIMENTS
        assert len(EXPERIMENTS) == 28

    def test_additional_experiments(self):
        """Test that ADDITIONAL_EXPERIMENTS contains expected items."""
        assert "igt" in ADDITIONAL_EXPERIMENTS


class TestMain:
    """Tests for main function."""

    @patch("experiments.verify_protocols.Path")
    def test_main_runs(self, mock_path, capsys):
        """Test that main function runs without errors."""
        mock_path.return_value.parent = Path("/fake/path")

        with patch("experiments.verify_protocols.verify_experiment") as mock_verify:
            mock_verify.return_value = VerificationResult(
                experiment="test", passed=["1. Test"], failed=[], warnings=[]
            )
            main()
            captured = capsys.readouterr()
            assert "APGI Experiment Protocol Verification Report" in captured.out

    @patch("experiments.verify_protocols.Path")
    def test_main_summary_output(self, mock_path, capsys):
        """Test main function prints summary correctly."""
        mock_path.return_value.parent = Path("/fake/path")

        with patch("experiments.verify_protocols.verify_experiment") as mock_verify:
            mock_verify.return_value = VerificationResult(
                experiment="test",
                passed=["1", "2", "3", "4", "5", "6"],
                failed=[],
                warnings=[],
            )
            main()
            captured = capsys.readouterr()
            assert "SUMMARY" in captured.out
            assert "Total experiments:" in captured.out

    @patch("experiments.verify_protocols.Path")
    def test_main_lists_incomplete(self, mock_path, capsys):
        """Test main lists incomplete experiments."""
        mock_path.return_value.parent = Path("/fake/path")

        with patch("experiments.verify_protocols.verify_experiment") as mock_verify:
            mock_verify.return_value = VerificationResult(
                experiment="incomplete",
                passed=["1", "2"],
                failed=["3", "4"],
                warnings=[],
            )
            main()
            captured = capsys.readouterr()
            assert "Experiments needing attention" in captured.out
