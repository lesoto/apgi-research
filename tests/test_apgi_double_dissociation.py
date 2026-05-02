"""
Comprehensive tests for apgi_double_dissociation.py - APGI Double Dissociation Protocol.
"""

from unittest.mock import MagicMock

import pytest

from apgi_double_dissociation import (
    DoubleDissociationProtocol,
    SessionData,
    automated_double_dissociation_task,
)


class TestSessionData:
    """Tests for SessionData dataclass."""

    def test_session_data_creation(self):
        """Test SessionData creation with all fields."""
        session = SessionData(
            session_id="session_001",
            heartbeat_accuracy=0.85,
            eeg_alpha_power=12.5,
            eeg_gamma_power=25.0,
            timestamp=1234567890.0,
        )
        assert session.session_id == "session_001"
        assert session.heartbeat_accuracy == 0.85
        assert session.eeg_alpha_power == 12.5
        assert session.eeg_gamma_power == 25.0
        assert session.timestamp == 1234567890.0

    def test_session_data_default_timestamp(self):
        """Test SessionData with default timestamp."""
        session = SessionData(
            session_id="session_002",
            heartbeat_accuracy=0.80,
            eeg_alpha_power=10.0,
            eeg_gamma_power=20.0,
        )
        assert session.timestamp == 0.0


class TestDoubleDissociationProtocolInitialization:
    """Tests for DoubleDissociationProtocol initialization."""

    def test_default_initialization(self):
        """Test default protocol initialization."""
        protocol = DoubleDissociationProtocol()
        assert protocol.min_sessions == 3
        assert protocol.target_icc == 0.65
        assert protocol.sessions == []
        assert protocol.pi_i_baseline is None
        assert protocol.beta_fitted is None

    def test_custom_initialization(self):
        """Test protocol with custom parameters."""
        protocol = DoubleDissociationProtocol(min_sessions=5, target_icc=0.70)
        assert protocol.min_sessions == 5
        assert protocol.target_icc == 0.70


class TestExtractEegPrior:
    """Tests for extract_eeg_prior method."""

    def test_extract_eeg_prior_normal(self):
        """Test EEG prior extraction with normal values."""
        protocol = DoubleDissociationProtocol()
        result = protocol.extract_eeg_prior(alpha_power=12.5, gamma_power=25.0)
        expected = 12.5 / 25.0
        assert result == expected

    def test_extract_eeg_prior_zero_gamma(self):
        """Test EEG prior with zero gamma power."""
        protocol = DoubleDissociationProtocol()
        result = protocol.extract_eeg_prior(alpha_power=12.5, gamma_power=0.0)
        assert result == 1.0  # Default fallback

    def test_extract_eeg_prior_negative_gamma(self):
        """Test EEG prior with negative gamma power."""
        protocol = DoubleDissociationProtocol()
        result = protocol.extract_eeg_prior(alpha_power=12.5, gamma_power=-5.0)
        assert result == 1.0  # Default fallback

    def test_extract_eeg_prior_high_alpha(self):
        """Test EEG prior with high alpha relative to gamma."""
        protocol = DoubleDissociationProtocol()
        result = protocol.extract_eeg_prior(alpha_power=50.0, gamma_power=10.0)
        assert result == 5.0


class TestValidateStage1Anchor:
    """Tests for validate_stage1_anchor method."""

    def test_insufficient_sessions(self):
        """Test validation with insufficient sessions."""
        protocol = DoubleDissociationProtocol()
        # Less than 3 sessions
        protocol.sessions = [
            SessionData("s1", 0.8, 10.0, 20.0),
        ]
        result = protocol.validate_stage1_anchor()
        assert result is False

    def test_exactly_min_sessions(self):
        """Test validation with exactly minimum sessions."""
        protocol = DoubleDissociationProtocol()
        protocol.sessions = [
            SessionData("s1", 0.8, 10.0, 20.0),
            SessionData("s2", 0.82, 11.0, 21.0),
            SessionData("s3", 0.81, 10.5, 20.5),
        ]
        result = protocol.validate_stage1_anchor()
        # Should pass with high accuracy and low variance
        assert result is True
        assert protocol.pi_i_baseline is not None

    def test_low_reliability(self):
        """Test validation with high variance (low reliability)."""
        protocol = DoubleDissociationProtocol()
        # High variance in accuracies
        protocol.sessions = [
            SessionData("s1", 0.3, 10.0, 20.0),
            SessionData("s2", 0.9, 11.0, 21.0),
            SessionData("s3", 0.1, 10.5, 20.5),
        ]
        result = protocol.validate_stage1_anchor()
        # High variance should cause low ICC
        assert result is False

    def test_many_sessions(self):
        """Test validation with many sessions."""
        protocol = DoubleDissociationProtocol()
        protocol.sessions = [
            SessionData(f"s{i}", 0.8 + (i * 0.01), 10.0 + i, 20.0 + i)
            for i in range(10)
        ]
        result = protocol.validate_stage1_anchor()
        assert result is True


class TestRunTwoStageEstimation:
    """Tests for run_two_stage_estimation method."""

    def test_stage2_without_stage1(self):
        """Test that Stage 2 fails without validated Stage 1."""
        protocol = DoubleDissociationProtocol()
        protocol.sessions = [
            SessionData("s1", 0.8, 10.0, 20.0),
            SessionData("s2", 0.82, 11.0, 21.0),
        ]
        # Less than 3 sessions, so Stage 1 won't validate
        with pytest.raises(ValueError, match="Stage 1 anchor not validated"):
            protocol.run_two_stage_estimation([])

    def test_stage2_with_valid_stage1(self):
        """Test Stage 2 estimation with valid Stage 1."""
        protocol = DoubleDissociationProtocol()
        # Set up valid Stage 1
        protocol.sessions = [
            SessionData("s1", 0.8, 10.0, 20.0),
            SessionData("s2", 0.82, 11.0, 21.0),
            SessionData("s3", 0.81, 10.5, 20.5),
        ]
        # Validate Stage 1 first
        assert protocol.validate_stage1_anchor() is True

        # Now run Stage 2
        trial_data = [
            {"somatic_marker": 0.5, "m_0": 0.0, "effective_precision": 0.9},
            {"somatic_marker": 0.3, "m_0": 0.0, "effective_precision": 0.85},
        ]
        result = protocol.run_two_stage_estimation(trial_data)
        assert "pi_i_baseline" in result
        assert "beta" in result
        assert result["pi_i_baseline"] is not None
        assert result["beta"] is not None

    def test_stage2_with_empty_trial_data(self):
        """Test Stage 2 with empty trial data."""
        protocol = DoubleDissociationProtocol()
        protocol.sessions = [
            SessionData("s1", 0.8, 10.0, 20.0),
            SessionData("s2", 0.82, 11.0, 21.0),
            SessionData("s3", 0.81, 10.5, 20.5),
        ]
        protocol.validate_stage1_anchor()

        result = protocol.run_two_stage_estimation([])
        assert result["pi_i_baseline"] is not None
        assert result["beta"] == 1.5  # Default fallback

    def test_stage2_beta_calculation(self):
        """Test beta calculation formula."""
        protocol = DoubleDissociationProtocol()
        protocol.sessions = [
            SessionData("s1", 0.8, 10.0, 20.0),
            SessionData("s2", 0.82, 11.0, 21.0),
            SessionData("s3", 0.81, 10.5, 20.5),
        ]
        protocol.validate_stage1_anchor()

        # Trial with m > m_0 should produce beta > 0
        trial_data = [
            {"somatic_marker": 1.0, "m_0": 0.0, "effective_precision": 0.9},
        ]
        result = protocol.run_two_stage_estimation(trial_data)
        beta = result.get("beta")
        assert beta is not None and beta > 0

    def test_stage2_with_zero_sigmoid(self):
        """Test Stage 2 when sigmoid approaches zero."""
        protocol = DoubleDissociationProtocol()
        protocol.sessions = [
            SessionData("s1", 0.8, 10.0, 20.0),
            SessionData("s2", 0.82, 11.0, 21.0),
            SessionData("s3", 0.81, 10.5, 20.5),
        ]
        protocol.validate_stage1_anchor()

        # When m is very negative, sigmoid approaches 0
        trial_data = [
            {"somatic_marker": -10.0, "m_0": 0.0, "effective_precision": 0.8},
        ]
        result = protocol.run_two_stage_estimation(trial_data)
        # Beta should still be calculated or use default
        assert result["beta"] is not None


class TestCheckDistributionDivergence:
    """Tests for check_distribution_divergence method."""

    def test_diverged_distributions(self):
        """Test when distributions have diverged."""
        protocol = DoubleDissociationProtocol()
        dist_beta = {"mean": 1.5, "var": 0.1}
        dist_pi = {"mean": 0.8, "var": 0.1}
        result = protocol.check_distribution_divergence(dist_beta, dist_pi)
        assert result is True

    def test_non_diverged_close_means(self):
        """Test when means are too close."""
        protocol = DoubleDissociationProtocol()
        dist_beta = {"mean": 1.0, "var": 0.1}
        dist_pi = {"mean": 1.02, "var": 0.1}
        result = protocol.check_distribution_divergence(dist_beta, dist_pi)
        assert result is False

    def test_non_diverged_high_variance_beta(self):
        """Test when beta distribution has high variance."""
        protocol = DoubleDissociationProtocol()
        dist_beta = {"mean": 1.5, "var": 0.6}
        dist_pi = {"mean": 0.8, "var": 0.1}
        result = protocol.check_distribution_divergence(dist_beta, dist_pi)
        assert result is False

    def test_non_diverged_high_variance_pi(self):
        """Test when pi distribution has high variance."""
        protocol = DoubleDissociationProtocol()
        dist_beta = {"mean": 1.5, "var": 0.1}
        dist_pi = {"mean": 0.8, "var": 0.6}
        result = protocol.check_distribution_divergence(dist_beta, dist_pi)
        assert result is False

    def test_boundary_mean_difference(self):
        """Test boundary case for mean difference."""
        protocol = DoubleDissociationProtocol()
        # Exactly at the threshold (0.05)
        dist_beta = {"mean": 1.0, "var": 0.1}
        dist_pi = {"mean": 1.05, "var": 0.1}
        result = protocol.check_distribution_divergence(dist_beta, dist_pi)
        # 1.05 - 1.0 = 0.05, which is not < 0.05, so should be True
        assert result is True


class TestComputeMockIcc:
    """Tests for _compute_mock_icc method."""

    def test_mock_icc_single_value(self):
        """Test ICC with single value."""
        protocol = DoubleDissociationProtocol()
        result = protocol._compute_mock_icc([0.8])
        assert result == 0.0

    def test_mock_icc_low_variance(self):
        """Test ICC with low variance (high reliability)."""
        protocol = DoubleDissociationProtocol()
        # Very consistent values
        data = [0.81, 0.82, 0.80, 0.81, 0.82]
        result = protocol._compute_mock_icc(data)
        # Low variance should give high ICC
        assert result > 0.65

    def test_mock_icc_high_variance(self):
        """Test ICC with high variance (low reliability)."""
        protocol = DoubleDissociationProtocol()
        # Very inconsistent values
        data = [0.3, 0.9, 0.1, 0.8, 0.2]
        result = protocol._compute_mock_icc(data)
        # High variance should give low ICC
        assert result < 0.65

    def test_mock_icc_clipping(self):
        """Test ICC value clipping."""
        protocol = DoubleDissociationProtocol()
        # Extreme values should be clipped to [0.4, 0.9]
        result = protocol._compute_mock_icc([0.8, 0.8, 0.8])
        assert 0.4 <= result <= 0.9


class TestAutomatedDoubleDissociationTask:
    """Tests for automated_double_dissociation_task function."""

    def test_function_returns_protocol(self):
        """Test that function returns a protocol instance."""
        mock_context = MagicMock()
        result = automated_double_dissociation_task(mock_context)
        assert isinstance(result, DoubleDissociationProtocol)

    def test_function_with_none_context(self):
        """Test function with None context."""
        result = automated_double_dissociation_task(None)
        assert isinstance(result, DoubleDissociationProtocol)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_negative_alpha_power(self):
        """Test EEG prior with negative alpha power."""
        protocol = DoubleDissociationProtocol()
        result = protocol.extract_eeg_prior(alpha_power=-10.0, gamma_power=20.0)
        assert result == -0.5  # Negative alpha is allowed

    def test_very_high_gamma(self):
        """Test EEG prior with very high gamma."""
        protocol = DoubleDissociationProtocol()
        result = protocol.extract_eeg_prior(alpha_power=10.0, gamma_power=1000.0)
        assert result == 0.01

    def test_session_data_edge_cases(self):
        """Test SessionData with edge case values."""
        session = SessionData(
            session_id="",
            heartbeat_accuracy=0.0,
            eeg_alpha_power=0.0,
            eeg_gamma_power=0.001,  # Very small but positive
        )
        assert session.session_id == ""
        assert session.heartbeat_accuracy == 0.0

    def test_protocol_with_many_sessions(self):
        """Test protocol with many sessions."""
        protocol = DoubleDissociationProtocol()
        protocol.sessions = [
            SessionData(f"s{i}", 0.75 + (i % 10) * 0.01, 10.0 + i, 20.0 + i)
            for i in range(100)
        ]
        result = protocol.validate_stage1_anchor()
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__])
