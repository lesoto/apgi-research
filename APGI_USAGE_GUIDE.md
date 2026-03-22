# APGI Auto-Improvement System - Complete Usage Guide

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Logic Components](#core-logic-components)
3. [Experiment Protocol Verification](#experiment-protocol-verification)
4. [Step-by-Step Usage Instructions](#step-by-step-usage-instructions)
5. [Available Experiments](#available-experiments)
6. [Configuration Parameters](#configuration-parameters)
7. [Output Interpretation](#output-interpretation)
8. [Troubleshooting](#troubleshooting)

---

## System Overview

The APGI system is a computational framework for modeling psychological states and cognitive processes using dynamical systems theory. The auto-improvement component enables AI agents to autonomously optimize experiment parameters.

### Key Features

- **51 Psychological States**: Complete library of psychological states with phenomenological profiles
- **Dynamical System Equations**: Full implementation of surprise, threshold, somatic marker, arousal, and precision dynamics
- **Measurement Proxies**: HEP amplitude, P3b latency, detection thresholds, and ignition duration
- **Neuromodulator Integration**: ACh, NE, DA, 5-HT, CRF mappings to APGI parameters
- **Psychiatric Profiles**: GAD, MDD, and Psychosis disorder modeling
- **29 Experiment Protocols**: Validated cognitive and psychological experiments

---

## Core Logic Components

### 1. Foundational Equations (`FoundationalEquations`)

```python
# Prediction Error: ε = o - ŷ
eps_e = prediction_error(observed, predicted)

# Precision: Π = 1/σ²
Pi = precision(0.25)  # Returns 4.0

# Z-score: z = (x - μ) / σ
z = z_score(1.5, 1.0, 0.5)  # Returns 1.0
```

### 2. Core Ignition System (`CoreIgnitionSystem`)

```python
# Accumulated Signal: S = ½Πᵉ(εᵉ)² + ½Πⁱ_eff(εⁱ)²
S = accumulated_signal(Pi_e, eps_e, Pi_i_eff, eps_i)

# Effective Interoceptive Precision: Πⁱ_eff = Πⁱ[1 + β·σ(M - M₀)]
Pi_i_eff = effective_interoceptive_precision(Pi_i, M, M_0, beta)

# Ignition Probability: P(ignite) = σ(α(S - θ))
P = ignition_probability(S, theta, alpha)
```

### 3. Dynamical System Equations (`DynamicalSystemEquations`)

| Variable | Equation | Time Constant |
|----------|----------|---------------|
| Surprise (S) | dS/dt = -τ_S⁻¹S + ½Πᵉ(εᵉ)² + ½Πⁱ_eff(εⁱ)² + σ_Sξ_S | τ_S: 0.2-0.5s |
| Threshold (θ) | dθ/dt = τ_θ⁻¹(θ₀(A) - θ) + γ_M M + λ S + σ_θ ξ_θ | τ_θ: 30s |
| Somatic Marker (M) | dM/dt = τ_M⁻¹(M*(εⁱ) - M) + γ_context C + σ_M ξ_M | τ_M: 1.5s |
| Arousal (A) | dA/dt = τ_A⁻¹(A_target - A) + σ_A ξ_A | τ_A: 0.2s |
| Precision (Π) | dΠ/dt = α_Π(Π* - Π) + σ_Π ξ_Π | Variable |

### 4. Parameter Ranges (Validated)

| Parameter | Valid Range | Default | Description |
|-----------|-------------|---------|-------------|
| τ_S | 0.2-0.5s | 0.35s | Surprise decay time constant |
| α | 3.0-8.0 | 5.5 | Sigmoid steepness |
| β_som | 0.5-2.5 | 1.5 | Somatic influence gain |
| θ_survival | 0.1-0.5 | 0.3 | Threshold for survival content |
| θ_neutral | 0.5-1.0 | 0.7 | Threshold for neutral content |

### 5. Measurement Equations (`MeasurementEquations`)

```python
# Heartbeat-Evoked Potential (HEP)
HEP = compute_HEP(Pi_i_eff, M_ca, beta)  # μV

# P3b Latency
P3b = compute_P3b_latency(S_t, theta_t, Pi_e)  # ms

# Detection Threshold (d')
d_prime = compute_detection_threshold(theta_t, content_domain, neuromodulators)

# Ignition Duration
duration = compute_ignition_duration(P_ignition, S_t)  # ms
```

### 6. State Library (51 States)

**Categories:**
- **Optimal Functioning** (4): flow, focus, serenity, mindfulness
- **Positive Affective** (7): amusement, joy, pride, romantic_love_early, romantic_love_sustained, gratitude, hope, optimism
- **Cognitive/Attentional** (8): curiosity, boredom, creativity, inspiration, hyperfocus, fatigue, decision_fatigue, mind_wandering
- **Aversive Affective** (7): fear, anxiety, anger, guilt, shame, loneliness, overwhelm
- **Pathological/Extreme** (7): depression, learned_helplessness, pessimistic_depression, panic, dissociation, depersonalization, derealization
- **Altered/Boundary** (6): awe, trance, mystical_experience, ego_dissolution, peak_experience, nostalgia
- **Transitional/Contextual** (6): confusion, frustration, anticipation, relief, surprise, disappointment
- **Unelaborated** (6): contentment, interest, calm, neutral, alert, reflective

---

## Experiment Protocol Verification

### Verification Criteria

All 29 experiments were verified against 6 criteria:

| Criterion | Description | Status |
|-----------|-------------|--------|
| File Structure | Both `prepare_*.py` and `run_*.py` exist | ✅ All Pass |
| READ-ONLY Designation | `prepare_*.py` marked READ-ONLY | ✅ All Pass |
| AGENT-EDITABLE Designation | `run_*.py` marked AGENT-EDITABLE | ✅ All Pass |
| Time Budget | 600 seconds (10 minutes) defined | ✅ All Pass |
| Primary Metrics | Output format: `primary_metric: <value>` | ✅ All Pass |
| Import Structure | Proper imports from prepare files | ✅ All Pass |

### Verification Summary

```
Total Experiments: 29
Fully Compliant: 29
Non-Compliant: 0
Compliance Rate: 100%
```

---

## Step-by-Step Usage Instructions

### Step 1: Environment Setup

```bash
# Navigate to the auto-improvement directory
cd /Users/lesoto/Sites/PYTHON/apgi-experiments/auto-improvement

# Ensure dependencies are installed
pip install numpy matplotlib dataclasses
```

### Step 2: Run a Single Experiment

```bash
# Run Iowa Gambling Task
uv run run_iowa_gambling_task.py

# Run Attentional Blink
uv run run_attentional_blink.py

# Run Visual Search
uv run run_visual_search.py
```

### Step 3: Run APGI System Demo

```bash
# Run complete APGI demonstration
python APGI_System.py
```

This will:
1. Validate parameter ranges
2. Initialize 51 psychological states
3. Run a 75-second simulation
4. Generate comprehensive visualizations
5. Save results to `apgi_complete_output/`

### Step 4: Analyze Results

Output files are saved to `apgi_complete_output/`:
- `complete_dashboard.png` - Comprehensive visualization
- `corrected_parameters.json` - Validated parameters
- `simulation_summary.json` - Summary statistics

### Step 5: Modify Experiments (Agent-Editable Files)

Edit `run_*.py` files to optimize parameters:

```python
# Example: Modifying Iowa Gambling Task
NUM_TRIALS_CONFIG = 150  # Increase trials
BASE_LEARNING_RATE = 0.08  # Adjust learning
EXPLORATION_PROB = 0.10  # Reduce exploration
```

**Important:** Never modify `prepare_*.py` files - they are READ-ONLY.

### Step 6: Run Autonomous Improvement Loop

```bash
# The system supports autonomous optimization
# Each run outputs: primary_metric: <value>
# Goal: Maximize the primary metric
# Time budget: 600 seconds per run
```

---

## Available Experiments

### Decision-Making Tasks

| Experiment | Primary Metric | Description |
|------------|----------------|-------------|
| Iowa Gambling Task | `net_score` | Decision-making under uncertainty |
| Go/No-Go | `d_prime` | Response inhibition |
| Stop Signal | `stop_signal_rt` | Inhibitory control |
| Simon Effect | `simon_effect_ms` | Stimulus-response compatibility |

### Attention Tasks

| Experiment | Primary Metric | Description |
|------------|----------------|-------------|
| Attentional Blink | `blink_magnitude` | Temporal attention limitation |
| Posner Cueing | `cueing_effect_ms` | Spatial attention orienting |
| Visual Search | `conjunction_slope` | Search efficiency |
| Change Blindness | `change_detection_rate` | Visual change detection |
| Inattentional Blindness | `detection_rate` | Unexpected stimulus detection |
| Navon Task | `global_local_bias` | Global vs local processing |

### Memory Tasks

| Experiment | Primary Metric | Description |
|------------|----------------|-------------|
| Dual N-Back | `d_prime` | Working memory updating |
| Sternberg Memory | `memory_scan_rate` | Memory scanning speed |
| Working Memory Span | `span_size` | Working memory capacity |
| DRM False Memory | `false_alarm_rate` | False memory creation |
| Serial Reaction Time | `sequence_learning` | Implicit sequence learning |
| Artificial Grammar Learning | `grammar_accuracy` | Implicit grammar learning |
| Probabilistic Category Learning | `category_accuracy` | Probabilistic learning |

### Interference Tasks

| Experiment | Primary Metric | Description |
|------------|----------------|-------------|
| Stroop Effect | `interference_effect_ms` | Cognitive interference |
| Eriksen Flanker | `flanker_effect_ms` | Response competition |
| Masking | `backward_masking_effect` | Visual masking |

### Perception Tasks

| Experiment | Primary Metric | Description |
|------------|----------------|-------------|
| Binocular Rivalry | `dominance_duration` | Perceptual alternation |
| Multisensory Integration | `integration_index` | Cross-modal integration |
| Time Estimation | `temporal_precision` | Time perception accuracy |

### Specialized Tasks

| Experiment | Primary Metric | Description |
|------------|----------------|-------------|
| Somatic Marker Priming | `priming_effect` | Somatic influence on decisions |
| Interoceptive Gating | `gating_ratio` | Interoceptive processing |
| Metabolic Cost | `metabolic_efficiency` | Energy efficiency |
| Virtual Navigation | `navigation_accuracy` | Spatial navigation |
| AI Benchmarking | `benchmark_score` | AI performance metrics |

---

## Configuration Parameters

### APGI Integration Parameters

Each experiment includes APGI parameters optimized for the task:

```python
APGI_PARAMS = {
    "experiment_name": "experiment_id",
    "enabled": True,
    "tau_s": 0.35,    # Surprise decay (0.2-0.5s)
    "beta": 1.5,      # Somatic gain (0.5-2.5)
    "theta_0": 0.5,   # Baseline threshold
    "alpha": 5.5,     # Sigmoid steepness (3-8)
}
```

### Task-Specific Optimizations

| Task Type | τ_S | β | θ₀ | α |
|-----------|-----|---|----|----|
| Fast temporal (AB) | 0.25 | 1.8 | 0.4 | 6.0 |
| Decision (IGT) | 0.40 | 2.0 | 0.4 | 5.0 |
| Interference (Stroop) | 0.30 | 1.6 | 0.35 | 6.0 |
| Memory (N-Back) | 0.35 | 1.4 | 0.45 | 5.5 |
| Perception (Search) | 0.35 | 1.3 | 0.5 | 5.0 |

---

## Output Interpretation

### Primary Metrics

Each experiment outputs a primary metric to maximize:

```
net_score: 8.0000
completion_time_s: 45.23
```

### Simulation Outputs

The APGI system generates comprehensive outputs:

```
✅ Simulation complete: 127 ignitions detected

Measurements:
  • HEP Amplitude: 4.23 μV
  • P3b Latency: 312.5 ms
  • Detection Threshold (d'): 2.15
  • Anxiety Index: 0.45
```

### Visualization Dashboard

The dashboard includes:
1. Core dynamics (S, θ, ignitions)
2. Measurement proxies (HEP, P3b)
3. Neuromodulator dynamics (ACh, NE, DA, 5-HT)
4. Domain-specific analysis (survival vs neutral)
5. Psychiatric profile comparison
6. State space trajectory
7. Precision expectation gap (anxiety index)

---

## Troubleshooting

### Common Issues

**Issue:** `ImportError: cannot import name 'ExperimentAPGIRunner'`
**Solution:** Ensure `experiment_apgi_integration.py` exists in the directory.

**Issue:** Parameter validation warnings
**Solution:** Parameters outside valid ranges are automatically clipped. Check the warning message for details.

**Issue:** `TIME_BUDGET exceeded`
**Solution:** Reduce `NUM_TRIALS_CONFIG` or optimize the simulation loop.

### Verification Commands

```bash
# Verify deck configurations
python prepare_iowa_gambling_task.py

# Verify stimulus sets
python prepare_attentional_blink.py

# Run full equation verification
python APGI_System.py
```

### Parameter Validation

```python
from APGI_System import APGIParameters

params = APGIParameters(tau_S=0.35, alpha=5.5, beta=1.5)
violations = params.validate()
if violations:
    print(f"Violations: {violations}")
```

---

## Summary

The APGI Auto-Improvement System provides:

- ✅ **29 validated experiment protocols** with 100% compliance
- ✅ **Complete dynamical system implementation** (51 states, all equations)
- ✅ **Validated parameter ranges** (τ_S: 0.2-0.5s, α: 3-8, β: 0.5-2.5)
- ✅ **Measurement equation outputs** (HEP, P3b, d', duration)
- ✅ **Neuromodulator integration** (ACh, NE, DA, 5-HT, CRF)
- ✅ **Psychiatric profile modeling** (GAD, MDD, Psychosis)
- ✅ **Autonomous optimization loop** with 600s time budget

For questions or issues, refer to the detailed documentation in `README.md` and `USAGE.md`.
