"""
Comprehensive tests for apgi_metrics.py - enhanced APGI metrics module.
"""

from apgi_metrics import (
    APGIMetricsSummary,
    EnhancedAPGIMetrics,
    IgnitionMetrics,
    MetabolicMetrics,
    SurpriseMetrics,
)


class TestIgnitionMetrics:
    """Tests for IgnitionMetrics dataclass."""

    def test_default_values(self):
        """Test default IgnitionMetrics values."""
        metrics = IgnitionMetrics(
            ignition_rate=0.5,
            mean_ignition_time=1.0,
            ignition_threshold=0.8,
            ignition_volatility=0.1,
            cumulative_ignition_prob=0.7,
        )
        assert metrics.ignition_rate == 0.5
        assert metrics.mean_ignition_time == 1.0
        assert metrics.ignition_threshold == 0.8
        assert metrics.ignition_volatility == 0.1
        assert metrics.cumulative_ignition_prob == 0.7
        assert metrics.ignition_events == []

    def test_with_ignition_events(self):
        """Test IgnitionMetrics with ignition events."""
        metrics = IgnitionMetrics(
            ignition_rate=0.5,
            mean_ignition_time=1.0,
            ignition_threshold=0.8,
            ignition_volatility=0.1,
            cumulative_ignition_prob=0.7,
            ignition_events=[1.0, 0.0, 1.0],
        )
        assert metrics.ignition_events == [1.0, 0.0, 1.0]


class TestSurpriseMetrics:
    """Tests for SurpriseMetrics dataclass."""

    def test_default_values(self):
        """Test default SurpriseMetrics values."""
        metrics = SurpriseMetrics(
            mean_surprise=0.5,
            surprise_variance=0.1,
            max_surprise=1.0,
            min_surprise=0.0,
            surprise_entropy=2.0,
            prediction_error_variance=0.15,
        )
        assert metrics.mean_surprise == 0.5
        assert metrics.surprise_variance == 0.1
        assert metrics.max_surprise == 1.0
        assert metrics.min_surprise == 0.0
        assert metrics.surprise_entropy == 2.0
        assert metrics.prediction_error_variance == 0.15


class TestMetabolicMetrics:
    """Tests for MetabolicMetrics dataclass."""

    def test_default_values(self):
        """Test default MetabolicMetrics values."""
        metrics = MetabolicMetrics(
            mean_metabolic_cost=0.5,
            total_metabolic_cost=10.0,
            metabolic_efficiency=0.8,
            cost_variance=0.1,
            optimal_cost_rate=0.9,
        )
        assert metrics.mean_metabolic_cost == 0.5
        assert metrics.total_metabolic_cost == 10.0
        assert metrics.metabolic_efficiency == 0.8
        assert metrics.cost_variance == 0.1
        assert metrics.optimal_cost_rate == 0.9


class TestAPGIMetricsSummary:
    """Tests for APGIMetricsSummary dataclass."""

    def test_default_values(self):
        """Test default APGIMetricsSummary values."""
        summary = APGIMetricsSummary()
        assert summary.ignition is None
        assert summary.surprise is None
        assert summary.metabolic is None
        assert summary.trial_count == 0
        assert summary.experiment_duration == 0.0
        assert summary.overall_performance_score == 0.0
        assert summary.statistical_significance is None

    def test_with_metrics(self):
        """Test APGIMetricsSummary with metrics."""
        ignition = IgnitionMetrics(0.5, 1.0, 0.8, 0.1, 0.7)
        surprise = SurpriseMetrics(0.5, 0.1, 1.0, 0.0, 2.0, 0.15)
        metabolic = MetabolicMetrics(0.5, 10.0, 0.8, 0.1, 0.9)

        summary = APGIMetricsSummary(
            ignition=ignition,
            surprise=surprise,
            metabolic=metabolic,
            trial_count=100,
            experiment_duration=300.0,
            overall_performance_score=0.85,
            statistical_significance={"test": 0.01},
        )
        assert summary.ignition == ignition
        assert summary.surprise == surprise
        assert summary.metabolic == metabolic
        assert summary.trial_count == 100
        assert summary.experiment_duration == 300.0
        assert summary.overall_performance_score == 0.85
        assert summary.statistical_significance == {"test": 0.01}


class TestEnhancedAPGIMetrics:
    """Tests for EnhancedAPGIMetrics class."""

    def test_initialization(self):
        """Test EnhancedAPGIMetrics initialization."""
        metrics = EnhancedAPGIMetrics()
        assert metrics.trial_data == []

    def test_calculate_ignition_metrics_empty(self):
        """Test calculate_ignition_metrics with empty data."""
        metrics = EnhancedAPGIMetrics()
        result = metrics.calculate_ignition_metrics([])
        assert result.ignition_rate == 0.0
        assert result.mean_ignition_time == 0.0
        assert result.ignition_threshold == 0.0
        assert result.ignition_volatility == 0.0
        assert result.cumulative_ignition_prob == 0.0
        assert result.ignition_events == []

    def test_calculate_ignition_metrics_basic(self):
        """Test calculate_ignition_metrics with basic data."""
        metrics = EnhancedAPGIMetrics()
        reaction_times = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = metrics.calculate_ignition_metrics(reaction_times)
        assert result.ignition_rate >= 0.0
        assert result.ignition_rate <= 1.0
        assert result.mean_ignition_time > 0.0
        assert result.ignition_threshold > 0.0
        assert len(result.ignition_events) == len(reaction_times)

    def test_calculate_ignition_metrics_with_threshold(self):
        """Test calculate_ignition_metrics with custom threshold."""
        metrics = EnhancedAPGIMetrics()
        reaction_times = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = metrics.calculate_ignition_metrics(reaction_times, threshold=0.7)
        assert result.ignition_threshold == 0.7
        # Events below threshold should be 1.0
        assert result.ignition_events[0] == 1.0  # 0.5 < 0.7
        assert result.ignition_events[4] == 0.0  # 0.9 > 0.7

    def test_calculate_ignition_metrics_with_predicted_times(self):
        """Test calculate_ignition_metrics with predicted times."""
        metrics = EnhancedAPGIMetrics()
        reaction_times = [0.5, 0.6, 0.7, 0.8, 0.9]
        predicted_times = [0.55, 0.65, 0.75, 0.85, 0.95]
        result = metrics.calculate_ignition_metrics(
            reaction_times, predicted_ignition_times=predicted_times
        )
        assert result.ignition_rate >= 0.0
        assert result.mean_ignition_time > 0.0

    def test_calculate_ignition_metrics_volatility(self):
        """Test ignition volatility calculation."""
        metrics = EnhancedAPGIMetrics()
        # High variability
        reaction_times = [0.1, 0.9, 0.2, 0.8, 0.3]
        result = metrics.calculate_ignition_metrics(reaction_times)
        assert result.ignition_volatility >= 0.0

    def test_calculate_ignition_metrics_cumulative_prob(self):
        """Test cumulative ignition probability calculation."""
        metrics = EnhancedAPGIMetrics()
        reaction_times = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = metrics.calculate_ignition_metrics(reaction_times)
        assert result.cumulative_ignition_prob >= 0.0
        assert result.cumulative_ignition_prob <= 1.0

    def test_calculate_surprise_metrics_empty(self):
        """Test calculate_surprise_metrics with empty data."""
        metrics = EnhancedAPGIMetrics()
        result = metrics.calculate_surprise_metrics([])
        assert result.mean_surprise == 0.0
        assert result.surprise_variance == 0.0
        assert result.max_surprise == 0.0
        assert result.min_surprise == 0.0
        assert result.surprise_entropy == 0.0
        assert result.prediction_error_variance == 0.0

    def test_calculate_surprise_metrics_basic(self):
        """Test calculate_surprise_metrics with basic data."""
        metrics = EnhancedAPGIMetrics()
        prediction_errors = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = metrics.calculate_surprise_metrics(prediction_errors)
        assert result.mean_surprise == 0.3
        assert result.surprise_variance > 0.0
        assert result.max_surprise == 0.5
        assert result.min_surprise == 0.1
        assert result.surprise_entropy >= 0.0

    def test_calculate_surprise_metrics_with_expected(self):
        """Test calculate_surprise_metrics with expected errors."""
        metrics = EnhancedAPGIMetrics()
        prediction_errors = [0.1, 0.2, 0.3, 0.4, 0.5]
        expected_errors = [0.15, 0.25, 0.35, 0.45, 0.55]
        result = metrics.calculate_surprise_metrics(
            prediction_errors, expected_errors=expected_errors
        )
        assert result.mean_surprise == 0.3
        assert result.prediction_error_variance > 0.0

    def test_calculate_surprise_metrics_entropy(self):
        """Test surprise entropy calculation."""
        metrics = EnhancedAPGIMetrics()
        # Uniform distribution
        prediction_errors = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = metrics.calculate_surprise_metrics(prediction_errors)
        assert result.surprise_entropy >= 0.0

    def test_calculate_metabolic_metrics_empty(self):
        """Test calculate_metabolic_metrics with empty data."""
        metrics = EnhancedAPGIMetrics()
        result = metrics.calculate_metabolic_metrics([])
        assert result.mean_metabolic_cost == 0.0
        assert result.total_metabolic_cost == 0.0
        assert result.metabolic_efficiency == 0.0
        assert result.cost_variance == 0.0
        assert result.optimal_cost_rate == 0.0

    def test_calculate_metabolic_metrics_basic(self):
        """Test calculate_metabolic_metrics with basic data."""
        metrics = EnhancedAPGIMetrics()
        reaction_times = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = metrics.calculate_metabolic_metrics(reaction_times)
        assert result.mean_metabolic_cost > 0.0
        assert result.total_metabolic_cost > 0.0
        assert result.metabolic_efficiency > 0.0
        assert result.cost_variance >= 0.0
        assert result.optimal_cost_rate > 0.0

    def test_calculate_metabolic_metrics_with_costs(self):
        """Test calculate_metabolic_metrics with provided costs."""
        metrics = EnhancedAPGIMetrics()
        reaction_times = [0.5, 0.6, 0.7, 0.8, 0.9]
        metabolic_costs = [1.0, 1.5, 2.0, 2.5, 3.0]
        result = metrics.calculate_metabolic_metrics(reaction_times, metabolic_costs)
        assert result.mean_metabolic_cost == 2.0
        assert result.total_metabolic_cost == 10.0

    def test_calculate_metabolic_metrics_with_time_budget(self):
        """Test calculate_metabolic_metrics with time budget."""
        metrics = EnhancedAPGIMetrics()
        reaction_times = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = metrics.calculate_metabolic_metrics(reaction_times, time_budget=10.0)
        assert result.metabolic_efficiency == 0.5  # 5 trials / 10 seconds

    def test_calculate_metabolic_metrics_efficiency_no_budget(self):
        """Test metabolic efficiency without time budget."""
        metrics = EnhancedAPGIMetrics()
        reaction_times = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = metrics.calculate_metabolic_metrics(reaction_times)
        # Efficiency = trials / sum of reaction times
        expected_efficiency = 5.0 / sum(reaction_times)
        assert abs(result.metabolic_efficiency - expected_efficiency) < 0.01

    def test_calculate_metabolic_metrics_optimal_cost_rate(self):
        """Test optimal cost rate calculation."""
        metrics = EnhancedAPGIMetrics()
        reaction_times = [0.5, 0.6, 0.7, 0.8, 0.9]
        metabolic_costs = [1.0, 1.5, 2.0, 2.5, 3.0]
        result = metrics.calculate_metabolic_metrics(reaction_times, metabolic_costs)
        # Optimal is 1.0, mean is 2.0, so rate should be 0.5
        assert result.optimal_cost_rate == 0.5

    def test_calculate_comprehensive_metrics_empty(self):
        """Test calculate_comprehensive_metrics with empty data."""
        metrics = EnhancedAPGIMetrics()
        result = metrics.calculate_comprehensive_metrics({})
        assert result.ignition is None
        assert result.surprise is None
        assert result.metabolic is None
        assert result.trial_count == 0
        assert result.experiment_duration == 0.0
        assert result.overall_performance_score == 0.0
        assert result.statistical_significance is None

    def test_calculate_comprehensive_metrics_full(self):
        """Test calculate_comprehensive_metrics with full data."""
        metrics = EnhancedAPGIMetrics()
        experiment_data = {
            "reaction_times": [0.5, 0.6, 0.7, 0.8, 0.9],
            "metabolic_costs": [1.0, 1.5, 2.0, 2.5, 3.0],
            "prediction_errors": [0.1, 0.2, 0.3, 0.4, 0.5],
            "predicted_ignition_times": [0.55, 0.65, 0.75, 0.85, 0.95],
            "time_budget": 10.0,
        }
        result = metrics.calculate_comprehensive_metrics(experiment_data)
        assert result.ignition is not None
        assert result.surprise is not None
        assert result.metabolic is not None
        assert result.trial_count == 5
        reaction_times: list[float] = experiment_data["reaction_times"]  # type: ignore[assignment]
        assert result.experiment_duration == sum(reaction_times)
        assert result.overall_performance_score >= 0.0

    def test_calculate_comprehensive_metrics_partial(self):
        """Test calculate_comprehensive_metrics with partial data."""
        metrics = EnhancedAPGIMetrics()
        experiment_data = {
            "reaction_times": [0.5, 0.6, 0.7],
        }
        result = metrics.calculate_comprehensive_metrics(experiment_data)
        assert result.ignition is not None
        assert result.surprise is None  # No prediction errors
        assert result.metabolic is not None
        assert result.trial_count == 3

    def test_calculate_comprehensive_metrics_performance_score(self):
        """Test overall performance score calculation."""
        metrics = EnhancedAPGIMetrics()
        experiment_data = {
            "reaction_times": [0.5, 0.6, 0.7, 0.8, 0.9],
            "metabolic_costs": [1.0, 1.5, 2.0, 2.5, 3.0],
            "time_budget": 10.0,
        }
        result = metrics.calculate_comprehensive_metrics(experiment_data)
        # Score should be between 0 and 1
        assert 0.0 <= result.overall_performance_score <= 1.0

    def test_calculate_comprehensive_metrics_statistical_significance(self):
        """Test statistical significance testing."""
        metrics = EnhancedAPGIMetrics()
        experiment_data = {
            "reaction_times": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
            "metabolic_costs": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
            "time_budget": 10.0,
        }
        result = metrics.calculate_comprehensive_metrics(experiment_data)
        assert result.statistical_significance is not None
        assert "ignition_vs_null" in result.statistical_significance
        assert "metabolic_vs_baseline" in result.statistical_significance

    def test_calculate_entropy_empty(self):
        """Test _calculate_entropy with empty values."""
        metrics = EnhancedAPGIMetrics()
        result = metrics._calculate_entropy([])
        assert result == 0.0

    def test_calculate_entropy_single_value(self):
        """Test _calculate_entropy with single value."""
        metrics = EnhancedAPGIMetrics()
        result = metrics._calculate_entropy([0.5])
        assert result >= 0.0

    def test_calculate_entropy_multiple_values(self):
        """Test _calculate_entropy with multiple values."""
        metrics = EnhancedAPGIMetrics()
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = metrics._calculate_entropy(values)
        assert result >= 0.0

    def test_calculate_entropy_uniform_distribution(self):
        """Test _calculate_entropy with uniform distribution."""
        metrics = EnhancedAPGIMetrics()
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        result = metrics._calculate_entropy(values)
        assert result >= 0.0

    def test_format_metrics_summary_empty(self):
        """Test format_metrics_summary with empty summary."""
        metrics = EnhancedAPGIMetrics()
        summary = APGIMetricsSummary()
        result = metrics.format_metrics_summary(summary)
        assert "ENHANCED APGI METRICS SUMMARY" in result
        assert "Trial Count: 0" in result

    def test_format_metrics_summary_full(self):
        """Test format_metrics_summary with full metrics."""
        metrics = EnhancedAPGIMetrics()
        ignition = IgnitionMetrics(0.5, 1.0, 0.8, 0.1, 0.7)
        surprise = SurpriseMetrics(0.5, 0.1, 1.0, 0.0, 2.0, 0.15)
        metabolic = MetabolicMetrics(0.5, 10.0, 0.8, 0.1, 0.9)
        summary = APGIMetricsSummary(
            ignition=ignition,
            surprise=surprise,
            metabolic=metabolic,
            trial_count=100,
            experiment_duration=300.0,
            overall_performance_score=0.85,
            statistical_significance={"test": 0.01},
        )
        result = metrics.format_metrics_summary(summary)
        assert "IGNITION METRICS" in result
        assert "SURPRISE METRICS" in result
        assert "METABOLIC METRICS" in result
        assert "EXPERIMENT OVERVIEW" in result
        assert "STATISTICAL SIGNIFICANCE" in result

    def test_format_metrics_summary_ignition_only(self):
        """Test format_metrics_summary with only ignition metrics."""
        metrics = EnhancedAPGIMetrics()
        ignition = IgnitionMetrics(0.5, 1.0, 0.8, 0.1, 0.7)
        summary = APGIMetricsSummary(ignition=ignition, trial_count=50)
        result = metrics.format_metrics_summary(summary)
        assert "IGNITION METRICS" in result
        assert "Ignition Rate: 0.500" in result

    def test_format_metrics_summary_significance_display(self):
        """Test statistical significance display in formatted summary."""
        metrics = EnhancedAPGIMetrics()
        summary = APGIMetricsSummary(
            statistical_significance={"test1": 0.01, "test2": 0.10}
        )
        result = metrics.format_metrics_summary(summary)
        assert "test1: p=0.0100" in result
        assert "✅ Significant" in result
        assert "test2: p=0.1000" in result
        assert "❌ Not Significant" in result


class TestEnhancedAPGIMetricsEdgeCases:
    """Tests for edge cases in EnhancedAPGIMetrics."""

    def test_single_reaction_time(self):
        """Test with single reaction time."""
        metrics = EnhancedAPGIMetrics()
        result = metrics.calculate_ignition_metrics([0.5])
        assert result.ignition_rate in [0.0, 1.0]
        assert result.mean_ignition_time == 0.5

    def test_all_fast_reactions(self):
        """Test with all fast reaction times."""
        metrics = EnhancedAPGIMetrics()
        reaction_times = [0.1, 0.15, 0.2, 0.25, 0.3]
        result = metrics.calculate_ignition_metrics(reaction_times, threshold=0.5)
        # All should be below threshold
        assert all(event == 1.0 for event in result.ignition_events)

    def test_all_slow_reactions(self):
        """Test with all slow reaction times."""
        metrics = EnhancedAPGIMetrics()
        reaction_times = [1.0, 1.5, 2.0, 2.5, 3.0]
        result = metrics.calculate_ignition_metrics(reaction_times, threshold=0.5)
        # All should be above threshold
        assert all(event == 0.0 for event in result.ignition_events)

    def test_zero_metabolic_cost(self):
        """Test with zero metabolic cost."""
        metrics = EnhancedAPGIMetrics()
        reaction_times = [0.5, 0.6, 0.7]
        metabolic_costs = [0.0, 0.0, 0.0]
        result = metrics.calculate_metabolic_metrics(reaction_times, metabolic_costs)
        assert result.mean_metabolic_cost == 0.0
        assert result.optimal_cost_rate == 1.0  # When mean is 0, defaults to 1.0

    def test_identical_prediction_errors(self):
        """Test with identical prediction errors (zero variance)."""
        metrics = EnhancedAPGIMetrics()
        prediction_errors = [0.5, 0.5, 0.5, 0.5, 0.5]
        result = metrics.calculate_surprise_metrics(prediction_errors)
        assert result.surprise_variance == 0.0
        assert result.max_surprise == 0.5
        assert result.min_surprise == 0.5

    def test_very_large_reaction_times(self):
        """Test with very large reaction times."""
        metrics = EnhancedAPGIMetrics()
        reaction_times = [100.0, 200.0, 300.0]
        result = metrics.calculate_metabolic_metrics(reaction_times)
        assert result.mean_metabolic_cost > 0.0
        assert result.metabolic_efficiency > 0.0

    def test_very_small_reaction_times(self):
        """Test with very small reaction times."""
        metrics = EnhancedAPGIMetrics()
        reaction_times = [0.001, 0.002, 0.003]
        result = metrics.calculate_metabolic_metrics(reaction_times)
        assert result.mean_metabolic_cost > 0.0
        # Should handle very small times without division by zero

    def test_negative_prediction_errors(self):
        """Test with negative prediction errors."""
        metrics = EnhancedAPGIMetrics()
        prediction_errors = [-0.5, -0.3, -0.1, 0.1, 0.3]
        result = metrics.calculate_surprise_metrics(prediction_errors)
        assert result.mean_surprise < 0.0
        assert result.min_surprise < 0.0

    def test_mixed_ignition_events(self):
        """Test with mixed ignition events."""
        metrics = EnhancedAPGIMetrics()
        reaction_times = [0.3, 0.7, 0.4, 0.8, 0.5]
        result = metrics.calculate_ignition_metrics(reaction_times, threshold=0.6)
        # Should have mix of 0 and 1 events
        assert 0.0 < result.ignition_rate < 1.0
        assert result.ignition_volatility > 0.0

    def test_no_time_budget_efficiency(self):
        """Test metabolic efficiency without time budget."""
        metrics = EnhancedAPGIMetrics()
        reaction_times = [1.0, 2.0, 3.0]
        result = metrics.calculate_metabolic_metrics(reaction_times, time_budget=None)
        assert result.metabolic_efficiency > 0.0
