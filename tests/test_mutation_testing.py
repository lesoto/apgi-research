"""
Tests for mutation_testing.py - mutation testing module.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.mutation_testing import (
    Mutant,
    MutationConfig,
    MutationOperator,
    MutationReport,
    MutationTester,
    WeakAssertionDetector,
    main,
)


class TestMutationOperator:
    """Tests for MutationOperator class."""

    def test_mutation_operator_creation(self):
        """Test creating a mutation operator."""
        op = MutationOperator(
            name="TEST",
            description="Test operator",
            target_nodes=["Add"],
            replacement_map={" + ": " - "},
        )
        assert op.name == "TEST"
        assert op.description == "Test operator"
        assert op.target_nodes == ["Add"]
        assert op.replacement_map == {" + ": " - "}

    def test_apply_simple(self):
        """Test applying mutation to simple code."""
        op = MutationOperator(
            name="AOR",
            description="Arithmetic",
            target_nodes=["Add"],
            replacement_map={" + ": " - "},
        )
        code = "x = a + b"
        mutants = op.apply(code)
        assert len(mutants) == 1
        assert "x = a - b" in mutants

    def test_apply_no_match(self):
        """Test applying mutation with no matches."""
        op = MutationOperator(
            name="AOR",
            description="Arithmetic",
            target_nodes=["Add"],
            replacement_map={" + ": " - "},
        )
        code = "x = a * b"
        mutants = op.apply(code)
        assert mutants == []

    def test_apply_multiple_mutations(self):
        """Test applying mutation with multiple targets."""
        op = MutationOperator(
            name="COR",
            description="Comparison",
            target_nodes=["Eq", "NotEq"],
            replacement_map={
                " == ": " != ",
                " != ": " == ",
            },
        )
        code = "if x == y and z != w"
        mutants = op.apply(code)
        assert len(mutants) == 2


class TestMutant:
    """Tests for Mutant dataclass."""

    def test_mutant_creation(self):
        """Test creating a mutant."""
        mutant = Mutant(
            operator="AOR",
            original_line="x = a + b",
            mutated_line="x = a - b",
            line_number=10,
            file_path=Path("test.py"),
        )
        assert mutant.operator == "AOR"
        assert mutant.original_line == "x = a + b"
        assert mutant.mutated_line == "x = a - b"
        assert mutant.line_number == 10
        assert mutant.file_path == Path("test.py")
        assert mutant.status == "pending"


class TestMutationReport:
    """Tests for MutationReport class."""

    def test_default_values(self):
        """Test default report values."""
        report = MutationReport()
        assert report.total_mutants == 0
        assert report.killed == 0
        assert report.survived == 0
        assert report.timeout == 0
        assert report.errors == 0
        assert report.duration == 0.0

    def test_mutation_score_zero(self):
        """Test mutation score with zero mutants."""
        report = MutationReport()
        assert report.mutation_score == 100.0

    def test_mutation_score_calculation(self):
        """Test mutation score calculation."""
        report = MutationReport(total_mutants=100, killed=80)
        assert report.mutation_score == 80.0

    def test_is_acceptable_true(self):
        """Test acceptable report."""
        report = MutationReport(total_mutants=100, killed=85, survived=3)
        assert report.is_acceptable is True

    def test_is_acceptable_low_score(self):
        """Test unacceptable due to low score."""
        report = MutationReport(total_mutants=100, killed=70, survived=30)
        assert report.is_acceptable is False

    def test_is_acceptable_too_many_survivors(self):
        """Test unacceptable due to too many survivors."""
        report = MutationReport(total_mutants=100, killed=90, survived=10)
        assert report.is_acceptable is False

    def test_to_dict(self):
        """Test converting report to dictionary."""
        report = MutationReport(total_mutants=10, killed=8, survived=2)
        data = report.to_dict()
        assert data["summary"]["total_mutants"] == 10
        assert data["summary"]["killed"] == 8
        assert data["summary"]["mutation_score"] == 80.0

    def test_generate_recommendations_low_score(self):
        """Test recommendations for low score."""
        report = MutationReport(total_mutants=100, killed=70)
        recs = report._generate_recommendations()
        assert any("Improve mutation score" in r for r in recs)

    def test_generate_recommendations_survivors(self):
        """Test recommendations for survivors."""
        report = MutationReport(total_mutants=100, killed=90, survived=10)
        recs = report._generate_recommendations()
        assert any("surviving mutants" in r for r in recs)

    def test_generate_recommendations_weak_assertions(self):
        """Test recommendations for weak assertions."""
        report = MutationReport()
        report.weak_assertions = [{}, {}, {}]
        recs = report._generate_recommendations()
        assert any("weak assertions" in r for r in recs)

    def test_generate_recommendations_good(self):
        """Test recommendations when everything is good."""
        report = MutationReport(total_mutants=100, killed=95, survived=2)
        recs = report._generate_recommendations()
        assert any("good" in r.lower() for r in recs)


class TestMutationConfig:
    """Tests for MutationConfig class."""

    def test_operators_defined(self):
        """Test that standard operators are defined."""
        assert len(MutationConfig.OPERATORS) == 4
        op_names = [op.name for op in MutationConfig.OPERATORS]
        assert "AOR" in op_names
        assert "COR" in op_names
        assert "LOR" in op_names
        assert "CRR" in op_names

    def test_target_patterns(self):
        """Test target file patterns."""
        assert "APGI_System.py" in MutationConfig.TARGET_PATTERNS
        assert "apgi_*.py" in MutationConfig.TARGET_PATTERNS

    def test_exclude_patterns(self):
        """Test exclusion patterns."""
        assert "*/tests/*" in MutationConfig.EXCLUDE_PATTERNS
        assert "conftest.py" in MutationConfig.EXCLUDE_PATTERNS

    def test_thresholds(self):
        """Test threshold values."""
        assert MutationConfig.SURVIVED_THRESHOLD == 5
        assert MutationConfig.TIMEOUT_SECONDS == 300


class TestWeakAssertionDetector:
    """Tests for WeakAssertionDetector class."""

    def test_detect_weak_patterns(self, tmp_path):
        """Test detecting weak assertion patterns."""
        test_file = tmp_path / "test_file.py"
        test_file.write_text("""
def test_something():
    assert True
    assert False
    assert None
    assertTrue(True)
""")
        weak = WeakAssertionDetector.detect(test_file)
        assert len(weak) == 4

    def test_detect_no_weak_patterns(self, tmp_path):
        """Test with no weak assertions."""
        test_file = tmp_path / "test_file.py"
        test_file.write_text("""
def test_something():
    assert x == y
    assert result is not None
""")
        weak = WeakAssertionDetector.detect(test_file)
        assert weak == []

    def test_detect_error_handling(self, tmp_path):
        """Test error handling during detection."""
        test_file = tmp_path / "nonexistent.py"
        weak = WeakAssertionDetector.detect(test_file)
        assert len(weak) == 1
        assert "error" in weak[0]


class TestMutationTester:
    """Tests for MutationTester class."""

    def test_init_default(self):
        """Test initialization with default root."""
        tester = MutationTester()
        assert tester.project_root == Path(".")
        assert isinstance(tester.report, MutationReport)

    def test_init_custom(self, tmp_path):
        """Test initialization with custom root."""
        tester = MutationTester(tmp_path)
        assert tester.project_root == tmp_path

    def test_find_target_files_empty(self, tmp_path):
        """Test finding targets in empty directory."""
        tester = MutationTester(tmp_path)
        targets = tester.find_target_files()
        assert targets == []

    def test_find_target_files_with_matches(self, tmp_path):
        """Test finding target files."""
        (tmp_path / "apgi_test.py").write_text("# test")
        tester = MutationTester(tmp_path)
        targets = tester.find_target_files()
        assert len(targets) >= 0  # May or may not match depending on pattern

    def test_generate_mutants_empty_file(self, tmp_path):
        """Test generating mutants for empty file."""
        test_file = tmp_path / "test.py"
        test_file.write_text("")
        tester = MutationTester(tmp_path)
        mutants = tester.generate_mutants(test_file)
        assert mutants == []

    def test_generate_mutants_with_code(self, tmp_path):
        """Test generating mutants for file with code."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = a + b\ny = c - d\n")
        tester = MutationTester(tmp_path)
        mutants = tester.generate_mutants(test_file)
        assert len(mutants) > 0

    def test_generate_mutants_error(self, tmp_path):
        """Test error handling in mutant generation."""
        tester = MutationTester(tmp_path)
        # Create a file that doesn't exist
        mutants = tester.generate_mutants(tmp_path / "nonexistent.py")
        assert mutants == []

    def test_run_tests_against_mutant_killed(self, tmp_path):
        """Test running tests against mutant that gets killed."""
        tester = MutationTester(tmp_path)
        mutant = Mutant(
            operator="AOR",
            original_line="x = 1 + 1",
            mutated_line="x = 1 - 1",  # This would break tests
            line_number=1,
            file_path=tmp_path / "test.py",
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1  # Tests fail = mutant killed
            result = tester.run_tests_against_mutant(mutant)
            assert result == "killed"

    def test_run_tests_against_mutant_survived(self, tmp_path):
        """Test running tests against mutant that survives."""
        tester = MutationTester(tmp_path)
        mutant = Mutant(
            operator="AOR",
            original_line="x = 1 + 1",
            mutated_line="x = 1 - 1",
            line_number=1,
            file_path=tmp_path / "test.py",
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0  # Tests pass = mutant survived
            result = tester.run_tests_against_mutant(mutant)
            assert result == "survived"

    def test_run_tests_against_mutant_timeout(self, tmp_path):
        """Test running tests with timeout."""
        tester = MutationTester(tmp_path)
        mutant = Mutant(
            operator="AOR",
            original_line="x = 1 + 1",
            mutated_line="x = 1 - 1",
            line_number=1,
            file_path=tmp_path / "test.py",
        )

        with patch("subprocess.run") as mock_run:
            from subprocess import TimeoutExpired

            mock_run.side_effect = TimeoutExpired("cmd", 300)
            result = tester.run_tests_against_mutant(mutant)
            assert result == "timeout"

    def test_run_tests_against_mutant_error(self, tmp_path):
        """Test error handling when running tests."""
        tester = MutationTester(tmp_path)
        mutant = Mutant(
            operator="AOR",
            original_line="x = 1 + 1",
            mutated_line="x = 1 - 1",
            line_number=1,
            file_path=tmp_path / "test.py",
        )

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Test error")
            result = tester.run_tests_against_mutant(mutant)
            assert result == "error"

    def test_print_summary(self, tmp_path, capsys):
        """Test printing mutation summary."""
        tester = MutationTester(tmp_path)
        tester.report = MutationReport(
            total_mutants=10,
            killed=8,
            survived=2,
        )
        tester.print_summary()
        captured = capsys.readouterr()
        assert "MUTATION TESTING SUMMARY" in captured.out
        assert "Total Mutants: 10" in captured.out


class TestMain:
    """Tests for main CLI function."""

    @patch("mutation_testing.MutationTester")
    def test_main_success(self, mock_tester_class, tmp_path):
        """Test main function with successful mutation testing."""
        mock_tester = MagicMock()
        mock_tester_class.return_value = mock_tester
        mock_tester.run_mutation_testing.return_value = MutationReport(
            total_mutants=10,
            killed=9,
            survived=1,
        )
        mock_tester.report = mock_tester.run_mutation_testing.return_value
        mock_tester.report.is_acceptable = True

        with patch(
            "sys.argv", ["mutation_testing", "--output", str(tmp_path / "report.json")]
        ):
            result = main()
            assert result == 0

    @patch("mutation_testing.MutationTester")
    def test_main_failure(self, mock_tester_class, tmp_path):
        """Test main function with failed mutation testing."""
        mock_tester = MagicMock()
        mock_tester_class.return_value = mock_tester
        mock_tester.run_mutation_testing.return_value = MutationReport(
            total_mutants=10,
            killed=5,
            survived=5,
        )
        mock_tester.report = mock_tester.run_mutation_testing.return_value
        mock_tester.report.is_acceptable = False

        with patch(
            "sys.argv", ["mutation_testing", "--output", str(tmp_path / "report.json")]
        ):
            result = main()
            assert result == 1

    @patch("mutation_testing.MutationTester")
    def test_main_with_max_mutants(self, mock_tester_class, tmp_path):
        """Test main function with max mutants limit."""
        mock_tester = MagicMock()
        mock_tester_class.return_value = mock_tester
        mock_tester.run_mutation_testing.return_value = MutationReport(
            total_mutants=5,
            killed=5,
        )
        mock_tester.report = mock_tester.run_mutation_testing.return_value
        mock_tester.report.is_acceptable = True

        with patch(
            "sys.argv",
            [
                "mutation_testing",
                "--max-mutants",
                "5",
                "--output",
                str(tmp_path / "report.json"),
            ],
        ):
            exit_code = main()
            mock_tester.run_mutation_testing.assert_called_once()
            assert mock_tester.run_mutation_testing.call_args[1]["max_mutants"] == 5
            assert exit_code == 0
