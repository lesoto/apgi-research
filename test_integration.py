"""
Integration tests for APGI experiments.

Tests the integration of all APGI components in realistic experiment scenarios.
"""

import pytest
import tempfile
from pathlib import Path
import time
import json
import numpy as np

# Import APGI modules
from apgi_integration import APGIIntegration
from validation import validate_modifications_before_apply
from git_operations import GitRollbackManager
from progress_tracking import ProgressTracker, TrialResult
from performance_monitoring import PerformanceMonitor


class TestExperimentIntegration:
    """Test cases for full experiment integration."""

    def test_complete_experiment_workflow(self):
        """Test a complete experiment workflow from start to finish."""
        # Initialize components
        apgi = APGIIntegration()
        progress_tracker = ProgressTracker("test_experiment", "test_participant", 10)
        performance_monitor = PerformanceMonitor()

        # Start performance monitoring
        performance_monitor.start_monitoring(interval=0.1)

        # Run experiment trials
        for trial in range(5):
            # Start trial
            progress_tracker.start_trial(trial + 1)

            # Start performance monitoring for trial
            metrics = performance_monitor.start_operation(f"trial_{trial + 1}")

            # Simulate trial data
            observed = np.random.normal(0.5, 0.2)
            predicted = np.random.normal(0.4, 0.15)
            trial_type = np.random.choice(["neutral", "survival", "visual"])

            # Process trial with APGI
            state = apgi.process_trial(observed, predicted, trial_type)

            # Record trial result
            trial_result = TrialResult(
                trial_number=trial + 1,
                timestamp=time.time(),
                response_time=np.random.normal(500, 100),
                accuracy=np.random.uniform(0.6, 0.95),
                apgi_metrics={
                    "ignition_prob": state["ignition_prob"],
                    "surprise": apgi.compute_surprise(observed, predicted),
                    "threshold": state["theta"],
                },
            )

            progress_tracker.complete_trial(trial_result)
            performance_monitor.end_operation(metrics, success=True)

            # Verify trial completion
            assert progress_tracker.progress.completed_trials == trial + 1
            assert len(progress_tracker.progress.trials) == trial + 1

        # Complete experiment
        progress_tracker.complete_experiment()
        performance_monitor.stop_monitoring()

        # Verify results
        assert progress_tracker.progress.status == "completed"
        assert progress_tracker.get_progress_percentage() == 50.0  # 5/10 trials

        # Check performance summary
        perf_summary = performance_monitor.get_performance_summary()
        assert perf_summary["total_operations"] == 5
        assert perf_summary["success_rate"] == 1.0

        # Check APGI summary
        apgi_summary = apgi.finalize()
        assert "ignition_rate" in apgi_summary
        assert "mean_surprise" in apgi_summary

    def test_experiment_with_validation(self):
        """Test experiment with validation checks."""
        # Initialize components
        apgi = APGIIntegration()
        progress_tracker = ProgressTracker(
            "validated_experiment", "test_participant", 5
        )

        # Validate experiment configuration
        config = {
            "experiment_name": "validated_experiment",
            "participant_id": "test_participant",
            "time_budget": 300.0,
            "trial_count": 5,
            "stimulus_type": "visual",
        }

        # This should pass validation
        from validation import validate_experiment_config

        validation_result = validate_experiment_config(config)
        assert validation_result.is_valid

        # Run trials with validation
        for trial in range(3):
            progress_tracker.start_trial(trial + 1)

            # Validate trial parameters
            trial_params = {
                "trial_number": trial + 1,
                "stimulus_intensity": np.random.uniform(0.1, 1.0),
                "response_timeout": 2000.0,
            }

            params_validation = validate_modifications_before_apply(trial_params)
            assert params_validation.is_valid

            # Process trial
            observed = np.random.normal(0.5, 0.2)
            predicted = np.random.normal(0.4, 0.15)
            state = apgi.process_trial(observed, predicted, "neutral")

            trial_result = TrialResult(
                trial_number=trial + 1,
                timestamp=time.time(),
                response_time=np.random.normal(500, 100),
                accuracy=np.random.uniform(0.7, 0.9),
                apgi_metrics={"ignition_prob": state["ignition_prob"]},
            )

            progress_tracker.complete_trial(trial_result)

        # Verify experiment completed successfully
        assert progress_tracker.progress.completed_trials == 3
        assert len(progress_tracker.progress.error_log) == 0

    def test_experiment_with_git_operations(self):
        """Test experiment with git operations and rollback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize git repository
            import subprocess

            subprocess.run(["git", "init"], cwd=temp_dir, check=True)
            subprocess.run(
                ["git", "config", "user.name", "Test"], cwd=temp_dir, check=True
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=temp_dir,
                check=True,
            )

            # Initialize components
            apgi = APGIIntegration()
            git_manager = GitRollbackManager(temp_dir)
            progress_tracker = ProgressTracker("git_experiment", "test_participant", 3)

            # Create experiment log file
            log_file = Path(temp_dir) / "experiment_log.json"

            # Run experiment with git tracking
            for trial in range(2):
                progress_tracker.start_trial(trial + 1)

                # Process trial
                observed = np.random.normal(0.5, 0.2)
                predicted = np.random.normal(0.4, 0.15)
                state = apgi.process_trial(observed, predicted, "neutral")

                trial_result = TrialResult(
                    trial_number=trial + 1,
                    timestamp=time.time(),
                    response_time=500.0,
                    accuracy=0.8,
                    apgi_metrics=state,
                )

                progress_tracker.complete_trial(trial_result)

                # Save experiment progress to file
                summary = progress_tracker.export_summary()
                log_file.write_text(json.dumps(summary, indent=2, default=str))

                # Stage file with git
                git_result = git_manager.stage_files(["experiment_log.json"])
                assert git_result.is_valid

                # Commit changes
                commit_result = git_manager.commit_changes(
                    f"Trial {trial + 1} completed"
                )
                assert commit_result.is_valid

            # Verify git operations
            git_status = git_manager.get_status()
            assert git_status["operations_count"] >= 2  # At least 2 commits

            # Test rollback capability
            rollback_result = git_manager.rollback_last_operation()
            assert rollback_result.is_valid

    def test_experiment_error_handling(self):
        """Test experiment error handling and recovery."""
        # Initialize components
        apgi = APGIIntegration()
        progress_tracker = ProgressTracker("error_experiment", "test_participant", 5)

        # Simulate error during trial
        for trial in range(3):
            progress_tracker.start_trial(trial + 1)

            try:
                if trial == 1:
                    # Simulate an error
                    raise ValueError("Simulated trial error")

                # Normal trial processing
                observed = np.random.normal(0.5, 0.2)
                predicted = np.random.normal(0.4, 0.15)
                state = apgi.process_trial(observed, predicted, "neutral")

                trial_result = TrialResult(
                    trial_number=trial + 1,
                    timestamp=time.time(),
                    response_time=500.0,
                    accuracy=0.8,
                    apgi_metrics=state,
                )

                progress_tracker.complete_trial(trial_result)

            except Exception as e:
                # Log error
                progress_tracker.log_error(f"Trial {trial + 1} error: {str(e)}")

                # Record failed trial
                trial_result = TrialResult(
                    trial_number=trial + 1, timestamp=time.time(), error=str(e)
                )

                progress_tracker.complete_trial(trial_result)

        # Verify error handling
        assert progress_tracker.progress.completed_trials == 3
        assert len(progress_tracker.progress.error_log) == 1
        assert "Simulated trial error" in progress_tracker.progress.error_log[0]

        # Check that some trials succeeded
        successful_trials = [
            t for t in progress_tracker.progress.trials if t.error is None
        ]
        assert len(successful_trials) == 2

    def test_experiment_performance_monitoring(self):
        """Test experiment with detailed performance monitoring."""
        # Initialize components
        apgi = APGIIntegration()
        progress_tracker = ProgressTracker("perf_experiment", "test_participant", 5)
        performance_monitor = PerformanceMonitor()

        # Add threshold callback
        violations = []

        def threshold_callback(violation_type, violation_data):
            violations.append((violation_type, violation_data))

        performance_monitor.add_threshold_callback(threshold_callback)

        # Start monitoring
        performance_monitor.start_monitoring(interval=0.05)

        # Run experiment with performance tracking
        for trial in range(3):
            # Monitor overall experiment operation
            exp_metrics = performance_monitor.start_operation("experiment_processing")

            progress_tracker.start_trial(trial + 1)

            # Monitor trial processing
            trial_metrics = performance_monitor.start_operation(
                f"trial_{trial + 1}_processing"
            )

            # Simulate some computation
            time.sleep(0.01)

            # Process with APGI
            observed = np.random.normal(0.5, 0.2)
            predicted = np.random.normal(0.4, 0.15)
            state = apgi.process_trial(observed, predicted, "neutral")

            trial_metrics.end_time = time.time()
            trial_metrics.duration = trial_metrics.end_time - trial_metrics.start_time
            trial_metrics.success = True

            performance_monitor.end_operation(trial_metrics, success=True)

            trial_result = TrialResult(
                trial_number=trial + 1,
                timestamp=time.time(),
                response_time=500.0,
                accuracy=0.8,
                apgi_metrics={"ignition_prob": state["ignition_prob"]},
            )

            progress_tracker.complete_trial(trial_result)

            exp_metrics.end_time = time.time()
            exp_metrics.duration = exp_metrics.end_time - exp_metrics.start_time
            exp_metrics.success = True

            performance_monitor.end_operation(exp_metrics, success=True)

        # Stop monitoring
        performance_monitor.stop_monitoring()

        # Verify performance data
        assert (
            len(performance_monitor.operation_metrics) >= 3
        )  # At least trial operations

        # Check performance trends
        memory_trend = performance_monitor.get_memory_trend()
        assert "current_mb" in memory_trend
        assert "trend_direction" in memory_trend

        cpu_trend = performance_monitor.get_cpu_trend()
        assert "current_percent" in cpu_trend
        assert "trend_direction" in cpu_trend

        # Check for performance regression
        regression = performance_monitor.detect_performance_regression()
        assert "status" in regression

    def test_experiment_data_persistence(self):
        """Test experiment data persistence and recovery."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize components
            apgi = APGIIntegration()
            progress_tracker = ProgressTracker(
                "persistent_experiment", "test_participant", 5, temp_dir
            )

            # Run some trials
            for trial in range(2):
                progress_tracker.start_trial(trial + 1)

                observed = np.random.normal(0.5, 0.2)
                predicted = np.random.normal(0.4, 0.15)
                state = apgi.process_trial(observed, predicted, "neutral")

                trial_result = TrialResult(
                    trial_number=trial + 1,
                    timestamp=time.time(),
                    response_time=500.0,
                    accuracy=0.8,
                    apgi_metrics=state,
                )

                progress_tracker.complete_trial(trial_result)

            # Save progress
            progress_tracker.save_summary_report()

            # Create new tracker and load progress
            new_tracker = ProgressTracker(
                "persistent_experiment",
                "test_participant",
                5,
                temp_dir,
                load_existing=True,
            )

            assert new_tracker.progress.completed_trials == 2
            assert len(new_tracker.progress.trials) == 2

            # Continue experiment
            for trial in range(2, 4):
                new_tracker.start_trial(trial + 1)

                observed = np.random.normal(0.5, 0.2)
                predicted = np.random.normal(0.4, 0.15)
                state = apgi.process_trial(observed, predicted, "survival")

                trial_result = TrialResult(
                    trial_number=trial + 1,
                    timestamp=time.time(),
                    response_time=500.0,
                    accuracy=0.8,
                    apgi_metrics=state,
                )

                new_tracker.complete_trial(trial_result)

            # Verify combined results
            assert new_tracker.progress.completed_trials == 4
            assert len(new_tracker.progress.trials) == 4

            # Check different trial types
            neutral_trials = [
                t
                for t in new_tracker.progress.trials
                if "survival" not in str(t.apgi_metrics)
            ]
            survival_trials = [
                t
                for t in new_tracker.progress.trials
                if "survival" in str(t.apgi_metrics)
            ]

            assert len(neutral_trials) == 2
            assert len(survival_trials) == 2


class TestMultiComponentIntegration:
    """Test cases for integration between multiple components."""

    def test_validation_and_progress_integration(self):
        """Test integration between validation and progress tracking."""
        progress_tracker = ProgressTracker("validated_progress", "test_participant", 5)

        # Test valid modifications during experiment
        valid_mods = {"time_budget": 300.0, "participant_id": "test_user"}
        validation_result = validate_modifications_before_apply(valid_mods)

        assert validation_result.is_valid
        progress_tracker.log_error("Validation passed for modifications")

        # Test invalid modifications
        invalid_mods = {"time_budget": -100.0}
        validation_result = validate_modifications_before_apply(invalid_mods)

        assert not validation_result.is_valid
        for error in validation_result.errors:
            progress_tracker.log_error(f"Validation error: {error}")

        assert len(progress_tracker.progress.error_log) >= 1

    def test_git_and_performance_integration(self):
        """Test integration between git operations and performance monitoring."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize git repository
            import subprocess

            subprocess.run(["git", "init"], cwd=temp_dir, check=True)
            subprocess.run(
                ["git", "config", "user.name", "Test"], cwd=temp_dir, check=True
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=temp_dir,
                check=True,
            )

            git_manager = GitRollbackManager(temp_dir)
            performance_monitor = PerformanceMonitor()

            # Monitor git operations
            git_metrics = performance_monitor.start_operation("git_operations")

            # Create and commit files
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test content")

            stage_result = git_manager.stage_files(["test.txt"])
            assert stage_result.is_valid

            commit_result = git_manager.commit_changes("Test commit")
            assert commit_result.is_valid

            # End performance monitoring
            performance_monitor.end_operation(git_metrics, success=True)

            # Verify performance data
            perf_summary = performance_monitor.get_performance_summary()
            assert perf_summary["total_operations"] == 1
            assert perf_summary["successful_operations"] == 1

            # Test rollback with performance monitoring
            rollback_metrics = performance_monitor.start_operation("git_rollback")
            rollback_result = git_manager.rollback_last_operation()
            performance_monitor.end_operation(rollback_metrics, success=True)

            assert rollback_result.is_valid

    def test_apgi_and_monitoring_integration(self):
        """Test integration between APGI and performance monitoring."""
        apgi = APGIIntegration()
        performance_monitor = PerformanceMonitor()

        # Monitor APGI operations
        for i in range(3):
            metrics = performance_monitor.start_operation(f"apgi_trial_{i + 1}")

            # Process trial with APGI
            observed = np.random.normal(0.5, 0.2)
            predicted = np.random.normal(0.4, 0.15)
            state = apgi.process_trial(observed, predicted, "neutral")

            # Simulate some processing time
            time.sleep(0.005)

            performance_monitor.end_operation(metrics, success=True)

            # Verify APGI state
            assert isinstance(state["ignition_prob"], (int, float))
            assert 0.0 <= state["ignition_prob"] <= 1.0

        # Get APGI summary
        apgi_summary = apgi.finalize()

        # Get performance summary
        perf_summary = performance_monitor.get_performance_summary()

        # Verify integration
        assert perf_summary["total_operations"] == 3
        assert perf_summary["success_rate"] == 1.0
        assert "ignition_rate" in apgi_summary


# Test configuration
if __name__ == "__main__":
    pytest.main([__file__])
