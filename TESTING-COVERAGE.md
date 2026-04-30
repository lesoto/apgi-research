# APGI API Test Coverage Analysis & 100% Coverage Roadmap

**Current Status:**

- **Total Test Files:** 45+ comprehensive test files
- **Core Module Coverage:** 65-95% (varies by module)
- **Experiment Files Coverage:** 0-90% (many at 0%)
- **Overall Coverage Target:** 100%

---

## Table of Contents

1. [Current Coverage Analysis](#1-current-coverage-analysis)
2. [Core Module Breakdown](#2-core-module-breakdown)
3. [Experiment Coverage Gap](#3-experiment-coverage-gap)
4. [100% Coverage Roadmap](#4-100-coverage-roadmap)
5. [Implementation Plan](#5-implementation-plan)
6. [Testing Strategy](#6-testing-strategy)
7. [Tools & Infrastructure](#7-tools--infrastructure)

---

## 1. Current Coverage Analysis

### 1.1 Test Suite Overview

The APGI testing framework comprises:

| Category | Count | Description |
|----------|-------|-------------|
| Unit Tests | 1039+ | Individual function/class tests |
| Integration Tests | 200+ | Cross-module interaction tests |
| Security Tests | 150+ | Vulnerability and security tests |
| Performance Tests | 100+ | Benchmarks and stress tests |
| E2E Tests | 50+ | Full system workflow tests |
| Mutation Tests | Configured | Test effectiveness validation |

### 1.2 Test File Inventory

#### Core Test Files (Comprehensive)

- `test_APGI_System.py` - Core APGI dynamical system tests
- `test_apgi_unit.py` - Unit tests for core functions
- `test_apgi_integration.py` - Integration tests
- `test_adversarial_apgi_core.py` - Adversarial edge case tests
- `test_apgi_validation.py` - Input validation tests
- `test_apgi_exception_paths.py` - Error handling tests
- `test_apgi_security_adapters.py` - Security adapter tests
- `test_apgi_timeout_abstraction.py` - Timeout/cross-platform tests
- `test_apgi_data_retention.py` - Data lifecycle tests
- `test_apgi_orchestration_kernel.py` - Orchestration tests

#### Specialized Test Files

- `test_xpr_agent_engine_comprehensive.py` - XPR agent engine (95 tests)
- `test_apgi_benchmarks.py` - Performance benchmarks
- `test_apgi_profiler.py` - Profiling tests
- `test_security.py` - Security vulnerability tests
- `test_security_controls.py` - Security control tests
- `test_autonomous_agent.py` - Autonomous agent tests
- `test_memory_store.py` - Memory system tests
- `test_memory_profiling.py` - Memory usage tests
- `test_gui_playwright.py` - GUI automation tests
- `test_rl_loop.py` - Reinforcement learning tests

#### Experiment Test Files

- `test_experiment_apgi_integration.py` - Experiment integration
- `test_run_change_blindness_full_apgi.py` - Change blindness
- `test_run_stroop_effect.py` - Stroop effect
- `test_standard_apgi_runner.py` - Standard runner
- `test_analyze_experiments.py` - Experiment analysis

---

## 2. Core Module Breakdown

### 2.1 High-Priority Core Modules

| Module | Lines | Current Coverage | Missing Lines | Priority |
|--------|-------|------------------|---------------|----------|
| APGI_System.py | 4026 | 85-90% | ~400 | Critical |
| apgi_integration.py | 1375 | 90% | ~135 | High |
| apgi_security.py | ~1500 | 80% | ~300 | Critical |
| apgi_validation.py | ~2000 | 85% | ~300 | High |
| apgi_config.py | ~1500 | 90% | ~150 | Medium |
| apgi_cli.py | ~800 | 70% | ~240 | Medium |
| autonomous_agent.py | ~1200 | 85% | ~180 | High |
| xpr_agent_engine.py | ~5000 | 75% | ~1250 | Critical |
| apgi_orchestration_kernel.py | ~2000 | 80% | ~400 | High |
| human_layer.py | ~1000 | 90% | ~100 | Medium |

### 2.2 APGI_System.py Coverage Details

**Module Purpose:** Core dynamical system implementation with 51 psychological states

**Key Components:**

1. FoundationalEquations class (prediction_error, z_score, precision)
2. SomaticMarkerSystem class (M_star computation, marker dynamics)
3. DynamicalSystem class (full simulation engine)
4. ParameterValidation class (range validation)
5. MeasurementEquations class (HEP, P3b, detection thresholds)
6. NeuromodulatorMapper class (ACh, NE, DA, 5-HT mapping)
7. PsychiatricProfiles class (GAD, MDD, Psychosis)
8. VisualizationEngine class (plotting functions)

**Coverage Gaps:**

- Exception handling paths (~150 lines)
- Visualization edge cases (~100 lines)
- Psychiatric profile boundary conditions (~80 lines)
- Measurement equation validation (~70 lines)

### 2.3 xpr_agent_engine.py Coverage Details

**Module Purpose:** XPR (eXperimental Personalization Runtime) agent engine

**Key Components:**

1. Agent lifecycle management
2. Memory systems (TF-IDF, embeddings)
3. LLM integration (LiteLLM)
4. Action execution
5. State management

**Coverage Gaps:**

- Error recovery paths (~400 lines)
- LLM fallback scenarios (~300 lines)
- Memory store edge cases (~250 lines)
- Concurrent access patterns (~150 lines)
- Token budget exhaustion (~150 lines)

---

## 3. Experiment Coverage Gap

### 3.1 Critical Finding: 40+ Experiment Files at 0% Coverage

The experiments directory contains 68 files with the following coverage distribution:

| Coverage Range | File Count | Description |
|----------------|------------|-------------|
| 0% | 40+ | No test coverage at all |
| 1-50% | 10 | Minimal coverage |
| 50-85% | 12 | Partial coverage |
| 85%+ | 6 | Good coverage |

### 3.2 Experiment Categories

#### Preparation Scripts (prepare_*.py) - 0% Coverage

- `prepare_change_blindness.py` - Change blindness paradigm setup
- `prepare_stroop_effect.py` - Stroop effect experiment setup
- `prepare_visual_search.py` - Visual search task setup
- `prepare_attentional_blink.py` - Attentional blink setup
- `prepare_dual_n_back.py` - Working memory task setup
- `prepare_go_no_go.py` - Inhibitory control setup
- `prepare_stop_signal.py` - Stop signal task setup
- `prepare_iowa_gambling_task.py` - Decision making setup
- `prepare_drm_false_memory.py` - False memory paradigm
- ... (25 more)

#### Runner Scripts (run_*.py) - 0-20% Coverage

- `run_change_blindness.py` - Execute change blindness
- `run_stroop_effect.py` - Execute Stroop task
- `run_visual_search.py` - Execute visual search
- `run_attentional_blink.py` - Execute attentional blink
- ... (30+ more)

### 3.3 Why Experiment Coverage Matters

1. **Regression Prevention:** Changes to APGI core may break experiments
2. **API Contract Validation:** Experiments validate the public API
3. **Documentation:** Tests serve as executable documentation
4. **Integration Validation:** End-to-end workflow verification

---

## 4. 100% Coverage Roadmap

### 4.1 Phase 1: Core Module Hardening (Week 1-2)

**Goal:** Achieve 95%+ coverage on all core modules

#### APGI_System.py (Target: 95%)

```python
# Required Tests:
- [ ] All FoundationalEquations edge cases (NaN, inf, zero)
- [ ] SomaticMarkerSystem boundary conditions
- [ ] DynamicalSystem simulation edge cases
- [ ] All exception handling paths
- [ ] ParameterValidation with extreme values
- [ ] MeasurementEquations with invalid inputs
- [ ] NeuromodulatorMapper boundary conditions
- [ ] PsychiatricProfiles validation
- [ ] VisualizationEngine error handling
- [ ] All static method combinations
```

#### xpr_agent_engine.py (Target: 95%)

```python
# Required Tests:
- [ ] Agent initialization edge cases
- [ ] Memory store failure modes
- [ ] LLM API error handling (all providers)
- [ ] Token budget exhaustion paths
- [ ] Concurrent access scenarios
- [ ] State persistence edge cases
- [ ] Action execution failures
- [ ] Embedding computation errors
- [ ] TF-IDF calculation edge cases
- [ ] All retry logic paths
```

#### apgi_security.py (Target: 95%)

```python
# Required Tests:
- [ ] All security adapter combinations
- [ ] Encryption/decryption edge cases
- [ ] Key rotation scenarios
- [ ] Audit log failure modes
- [ ] Permission validation paths
- [ ] All sanitization functions
```

### 4.2 Phase 2: Experiment Test Infrastructure (Week 2-3)

**Goal:** Create reusable experiment testing framework

#### Experiment Test Framework Components

```python
# experiments/test_framework.py
class ExperimentTestHarness:
    """Reusable test harness for all experiments."""
    
    def test_prepare_script_structure(self, prepare_module):
        """Verify all prepare scripts have required functions."""
        # Check for: generate_trials(), get_experiment_config()
        
    def test_runner_script_structure(self, runner_module):
        """Verify all runner scripts have required functions."""
        # Check for: run_experiment(), save_results()
        
    def test_json_output_validity(self, results):
        """Verify experiment output matches schema."""
        
    def test_apgi_metrics_integration(self, results):
        """Verify APGI metrics are properly computed."""
        
    def test_visualization_generation(self, module):
        """Verify plots can be generated without errors."""
```

#### Per-Experiment Test Requirements

Each experiment needs tests for:

1. **Configuration Validation** - All config parameters are valid
2. **Trial Generation** - Trials are generated correctly
3. **APGI Integration** - APGI metrics are computed
4. **Output Schema** - Results match expected format
5. **Visualization** - Plots can be generated
6. **Edge Cases** - Empty inputs, boundary values
7. **Error Handling** - Graceful failure modes

### 4.3 Phase 3: Experiment Coverage Implementation (Week 3-5)

**Goal:** Achieve 90%+ coverage on all experiment files

#### Priority 1: High-Impact Experiments (10 files)
1. `prepare_change_blindness.py` + `run_change_blindness.py`
2. `prepare_stroop_effect.py` + `run_stroop_effect.py`
3. `prepare_visual_search.py` + `run_visual_search.py`
4. `prepare_attentional_blink.py` + `run_attentional_blink.py`
5. `prepare_go_no_go.py` + `run_go_no_go.py`

#### Priority 2: Medium-Impact Experiments (20 files)
- Working memory experiments (dual_n_back, sternberg_memory)
- Decision making (iowa_gambling_task, probabilistic_category_learning)
- Attention experiments (posner_cueing, flanker)
- Memory experiments (drm_false_memory)
- Motor control (simon_effect, serial_reaction_time)

#### Priority 3: Remaining Experiments (38 files)
- All other prepare/run scripts
- Utility scripts (verify_protocols, migrate_runners)

### 4.4 Phase 4: Edge Case & Integration Coverage (Week 5-6)

**Goal:** Achieve 100% coverage on all remaining gaps

#### Edge Case Categories:
1. **Numeric Edge Cases**
   - NaN, inf, -inf handling
   - Very large/small values
   - Floating point precision limits
   - Division by zero scenarios

2. **Input Validation Edge Cases**
   - Empty collections
   - None/null handling
   - Type coercion edge cases
   - Unicode/string edge cases

3. **Concurrency Edge Cases**
   - Race conditions
   - Deadlock scenarios
   - Resource exhaustion
   - Timeout scenarios

4. **Security Edge Cases**
   - Injection attempts
   - Path traversal
   - Buffer overflow patterns
   - Cryptographic edge cases

---

## 5. Implementation Plan

### 5.1 Week-by-Week Breakdown

#### Week 1: Core Module Hardening - Part 1

##### Week 1 Tasks

- [ ] Complete APGI_System.py coverage gaps (400 lines)
  - [ ] Add 50 tests for FoundationalEquations edge cases
  - [ ] Add 40 tests for exception handling paths
  - [ ] Add 30 tests for visualization edge cases
  - [ ] Add 30 tests for psychiatric profile boundaries
  - [ ] Add 25 tests for measurement validation
  - [ ] Add 25 tests for neuromodulator boundaries
  
##### Week 1 Deliverable

APGI_System.py at 95% coverage

#### Week 2: Core Module Hardening - Part 2

##### Week 2 Tasks

- [ ] Complete xpr_agent_engine.py coverage (1250 lines)
  - [ ] Add 80 tests for error recovery paths
  - [ ] Add 60 tests for LLM fallback scenarios
  - [ ] Add 50 tests for memory store edge cases
  - [ ] Add 30 tests for concurrent access
  - [ ] Add 30 tests for token budget exhaustion
  
- [ ] Complete apgi_security.py coverage (300 lines)
  - [ ] Add 40 tests for security adapters
  - [ ] Add 30 tests for encryption edge cases
  - [ ] Add 20 tests for audit log failures
  - [ ] Add 10 tests for permission validation
  
- [ ] Complete remaining core modules
  - [ ] apgi_validation.py: 50 tests
  - [ ] apgi_orchestration_kernel.py: 40 tests
  - [ ] autonomous_agent.py: 30 tests
  - [ ] apgi_cli.py: 48 tests

##### Week 2 Deliverable

All core modules at 95%+ coverage

#### Week 3: Experiment Framework & Priority 1

##### Week 3 Tasks

- [ ] Create `experiments/test_framework.py`
  - [ ] Base test harness class
  - [ ] Mock APGI integration
  - [ ] Output validation utilities
  - [ ] Visualization test helpers
  
- [ ] Implement tests for Priority 1 experiments
  - [ ] change_blindness: 30 tests
  - [ ] stroop_effect: 30 tests
  - [ ] visual_search: 30 tests
  - [ ] attentional_blink: 30 tests
  - [ ] go_no_go: 30 tests

##### Week 3 Deliverable

Experiment framework + 5 fully tested experiments

#### Week 4: Priority 2 Experiments

##### Week 4 Tasks

- [ ] Working memory experiments
  - [ ] dual_n_back: 25 tests
  - [ ] sternberg_memory: 25 tests
  - [ ] working_memory_span: 25 tests
  
- [ ] Decision making experiments
  - [ ] iowa_gambling_task: 25 tests
  - [ ] probabilistic_category_learning: 25 tests
  
- [ ] Attention experiments
  - [ ] posner_cueing: 25 tests
  - [ ] eriksen_flanker: 25 tests
  - [ ] binocular_rivalry: 25 tests

##### Week 4 Deliverable

10 additional experiments at 90% coverage

#### Week 5: Priority 3 Experiments

##### Week 5 Tasks

- [ ] Memory experiments (3 files)
- [ ] Motor control experiments (3 files)
- [ ] Interoception experiments (3 files)
- [ ] Learning experiments (4 files)
- [ ] Remaining experiments (25 files)

##### Week 5 Deliverable

All 68 experiment files at 90%+ coverage

#### Week 6: Final Coverage Push

##### Week 6 Tasks

- [ ] Identify all remaining uncovered lines
- [ ] Add edge case tests for numeric boundaries
- [ ] Add security-focused adversarial tests
- [ ] Add performance/stress tests for critical paths
- [ ] Run full mutation testing
- [ ] Validate 100% coverage

##### Week 6 Deliverable

100% test coverage achieved

### 5.2 Resource Requirements

#### Estimated Test Counts by Phase

| Phase | New Tests | Cumulative Tests |
|-------|-----------|------------------|
| Week 1 | 200 | 1239 |
| Week 2 | 358 | 1597 |
| Week 3 | 180 | 1777 |
| Week 4 | 225 | 2002 |
| Week 5 | 400 | 2402 |
| Week 6 | 150 | 2552 |

**Total New Tests:** ~1,500 tests
**Total Tests After Completion:** ~2,500 tests

---

## 6. Testing Strategy

### 6.1 Test Categories for 100% Coverage

#### 1. Unit Tests (40% of new tests)
```python
# Example: FoundationalEquations edge case test
@pytest.mark.unit
@pytest.mark.boundary
def test_prediction_error_with_infinity():
    """Test prediction error handles infinity correctly."""
    result = FoundationalEquations.prediction_error(float('inf'), 1.0)
    assert result == float('inf')
    
    result = FoundationalEquations.prediction_error(1.0, float('inf'))
    assert result == float('-inf')
```

#### 2. Integration Tests (20% of new tests)
```python
# Example: Experiment with APGI integration
@pytest.mark.integration
def test_change_blindness_with_apgi_metrics():
    """Test change blindness experiment computes APGI metrics correctly."""
    from experiments.prepare_change_blindness import generate_trials
    from apgi_integration import APGIIntegration
    
    trials = generate_trials(n_trials=10)
    apgi = APGIIntegration()
    
    for trial in trials:
        metrics = apgi.compute_trial_metrics(trial)
        assert 'ignition_probability' in metrics
        assert 0 <= metrics['ignition_probability'] <= 1
```

#### 3. Adversarial Tests (15% of new tests)
```python
# Example: Security and edge case testing
@pytest.mark.adversarial
def test_xpr_agent_with_malicious_input():
    """Test XPR agent handles adversarial inputs safely."""
    agent = XPRAgentEngine()
    
    malicious_inputs = [
        "<script>alert('xss')</script>",
        "../../../etc/passwd",
        "\x00" * 1000,  # Null bytes
        "\u0000",  # Unicode null
    ]
    
    for inp in malicious_inputs:
        result = agent.process_input(inp)
        assert result.is_safe  # No exceptions, sanitized output
```

#### 4. Property-Based Tests (10% of new tests)
```python
# Example: Hypothesis-based testing
from hypothesis import given, strategies as st

@pytest.mark.property_based
@given(st.floats(min_value=-1e10, max_value=1e10),
       st.floats(min_value=-1e10, max_value=1e10))
def test_precision_is_positive(a, b):
    """Precision should always be positive."""
    result = FoundationalEquations.precision(a, b)
    assert result > 0 or np.isnan(result)
```

#### 5. Performance Tests (10% of new tests)
```python
# Example: Performance regression test
@pytest.mark.performance
def test_apgi_simulation_performance():
    """Test APGI simulation meets performance targets."""
    import time
    
    ds = DynamicalSystem()
    start = time.time()
    
    # Run 1000 iterations
    for _ in range(1000):
        ds.step()
    
    elapsed = time.time() - start
    assert elapsed < 1.0  # Must complete in under 1 second
```

#### 6. Mutation-Resistant Tests (5% of new tests)
```python
# Example: Tests that catch code mutations
@pytest.mark.mutation_resistant
def test_z_score_with_known_values():
    """Test z-score with mathematically verifiable results."""
    # These values are chosen to catch arithmetic mutations
    result = FoundationalEquations.z_score(10, 5, 2.5)
    assert result == 2.0  # (10-5)/2.5 = 2
    
    result = FoundationalEquations.z_score(0, 0, 1)
    assert result == 0.0
```

### 6.2 Test Naming Convention

```
test_<module>_<function>_<scenario>_<condition>

test_apgi_system_precision_with_infinity_returns_capped_value
test_xpr_engine_init_with_invalid_api_key_raises_authentication_error
test_change_blindness_prepare_with_zero_trials_returns_empty_list
test_security_adapter_encrypt_with_unicode_handles_utf8_correctly
```

### 6.3 Test Organization

```
tests/
├── unit/                    # Unit tests (mirrors src structure)
│   ├── test_apgi_system_unit.py
│   ├── test_apgi_integration_unit.py
│   └── ...
├── integration/             # Integration tests
│   ├── test_experiment_integration.py
│   └── ...
├── experiments/             # Experiment-specific tests
│   ├── test_change_blindness.py
│   ├── test_stroop_effect.py
│   └── ...
├── security/                # Security tests
│   └── ...
├── performance/             # Performance tests
│   └── ...
├── adversarial/             # Edge case tests
│   └── ...
├── conftest.py              # Shared fixtures
└── coverage_config.py       # Coverage configuration
```

---

## 7. Tools & Infrastructure

### 7.1 Coverage Tools

```bash
# Run full test suite with coverage
python -m pytest tests/ --cov=. --cov-report=html --cov-report=xml --cov-report=json

# Run with branch coverage
python -m pytest tests/ --cov=. --cov-branch --cov-report=term-missing

# Check coverage fails under threshold
python -m pytest tests/ --cov=. --cov-fail-under=100

# Generate coverage gaps report
python tests/coverage_config.py

# Run mutation testing
python tests/mutation_testing.py --max-mutants 100
```

### 7.2 Coverage Configuration

```ini
# .coveragerc
[run]
source = .
branch = True
parallel = True
concurrency = thread, multiprocessing
omit = 
    */tests/*
    */test_*
    */__pycache__/*
    */venv/*
    */.venv/*

[report]
fail_under = 100
skip_covered = False
show_missing = True
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
    @abstract
    @abc.abstractmethod

[html]
directory = htmlcov

[xml]
output = coverage.xml

[json]
output = coverage.json
```

### 7.3 CI/CD Integration

```yaml
# .github/workflows/coverage.yml
name: 100% Coverage Check

on: [push, pull_request]

jobs:
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          
      - name: Install dependencies
        run: pip install -e ".[test]"
        
      - name: Run tests with coverage
        run: |
          python -m pytest tests/ \
            --cov=. \
            --cov-branch \
            --cov-fail-under=100 \
            --cov-report=xml \
            --cov-report=html
            
      - name: Generate coverage gaps report
        run: |
          python -c "
          from tests.coverage_config import analyze_coverage_report
          from pathlib import Path
          report = analyze_coverage_report(Path('coverage.json'))
          print(report)
          "
          
      - name: Upload coverage report
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: htmlcov/
          
      - name: Mutation testing
        run: python tests/mutation_testing.py --max-mutants 50
```

### 7.4 Coverage Monitoring Dashboard

```python
# tests/coverage_dashboard.py
"""Real-time coverage monitoring dashboard."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

@dataclass
class CoverageMetrics:
    line_coverage: float
    branch_coverage: float
    missing_lines: int
    total_lines: int
    
class CoverageDashboard:
    """Monitor and visualize coverage progress."""
    
    def __init__(self, coverage_json_path: Path = Path("coverage.json")):
        self.coverage_path = coverage_json_path
        self.history: List[CoverageMetrics] = []
        
    def load_current(self) -> CoverageMetrics:
        """Load current coverage metrics."""
        with open(self.coverage_path) as f:
            data = json.load(f)
        
        totals = data.get("totals", {})
        return CoverageMetrics(
            line_coverage=totals.get("percent_covered", 0),
            branch_coverage=totals.get("percent_covered_branches", 0),
            missing_lines=totals.get("missing_lines", 0),
            total_lines=totals.get("num_statements", 0),
        )
    
    def check_progress(self) -> Dict:
        """Check progress toward 100% coverage."""
        current = self.load_current()
        
        return {
            "line_coverage_pct": current.line_coverage,
            "branch_coverage_pct": current.branch_coverage,
            "lines_remaining": current.missing_lines,
            "progress_pct": (current.total_lines - current.missing_lines) / current.total_lines * 100,
            "target_met": current.line_coverage >= 100 and current.branch_coverage >= 100,
        }
```

---

## Appendix A: Current Coverage Detailed Breakdown

### Core Module Line Counts

| Module | Total Lines | Covered | Missing | Coverage % |
|--------|-------------|---------|---------|------------|
| APGI_System.py | 4026 | ~3400 | ~626 | ~84% |
| xpr_agent_engine.py | ~5000 | ~3750 | ~1250 | ~75% |
| apgi_integration.py | 1375 | ~1238 | ~137 | ~90% |
| apgi_security.py | ~1500 | ~1200 | ~300 | ~80% |
| apgi_validation.py | ~2000 | ~1700 | ~300 | ~85% |
| apgi_config.py | ~1500 | ~1350 | ~150 | ~90% |
| apgi_cli.py | ~800 | ~560 | ~240 | ~70% |
| autonomous_agent.py | ~1200 | ~1020 | ~180 | ~85% |
| apgi_orchestration_kernel.py | ~2000 | ~1600 | ~400 | ~80% |
| apgi_profiler.py | ~1000 | ~850 | ~150 | ~85% |
| apgi_authz.py | ~800 | ~680 | ~120 | ~85% |
| apgi_data_retention.py | ~1500 | ~1350 | ~150 | ~90% |
| apgi_timeout_abstraction.py | ~500 | ~450 | ~50 | ~90% |
| human_layer.py | ~1000 | ~900 | ~100 | ~90% |

### Experiment Files Coverage

| Category | Files | Total Lines | Coverage % |
|----------|-------|-------------|------------|
| prepare_*.py | 34 | ~8,000 | 0-15% |
| run_*.py | 34 | ~12,000 | 0-20% |
| Total | 68 | ~20,000 | ~5% |

---

## Appendix B: Test Template for New Coverage

```python
"""Test template for achieving coverage on [MODULE_NAME]."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

# Import the module under test
from [module_path] import [ClassName], [function_name]


class Test[ClassName]Coverage:
    """Comprehensive tests for [ClassName] to achieve 100% coverage."""
    
    # =========================================================================
    # Happy Path Tests (Baseline Functionality)
    # =========================================================================
    
    def test_[method]_with_valid_input(self):
        """Test [method] with standard valid input."""
        obj = [ClassName]()
        result = obj.[method](valid_input)
        assert result is not None
        
    # =========================================================================
    # Boundary Value Tests
    # =========================================================================
    
    @pytest.mark.boundary
    def test_[method]_with_zero(self):
        """Test [method] with zero value."""
        obj = [ClassName]()
        result = obj.[method](0)
        assert isinstance(result, (int, float))
        
    @pytest.mark.boundary
    def test_[method]_with_negative(self):
        """Test [method] with negative value."""
        obj = [ClassName]()
        result = obj.[method](-1)
        assert result is not None
        
    @pytest.mark.boundary
    def test_[method]_with_max_float(self):
        """Test [method] with maximum float value."""
        obj = [ClassName]()
        result = obj.[method](float('1e308'))
        assert result is not None
        
    # =========================================================================
    # Exception Handling Tests
    # =========================================================================
    
    @pytest.mark.exception
    def test_[method]_with_none_raises_error(self):
        """Test [method] raises appropriate error for None input."""
        obj = [ClassName]()
        with pytest.raises((TypeError, ValueError)):
            obj.[method](None)
            
    @pytest.mark.exception
    def test_[method]_with_invalid_type_raises_error(self):
        """Test [method] raises error for invalid type."""
        obj = [ClassName]()
        with pytest.raises((TypeError, ValueError)):
            obj.[method]("invalid")
            
    # =========================================================================
    # Special Value Tests
    # =========================================================================
    
    @pytest.mark.special_values
    def test_[method]_with_nan(self):
        """Test [method] handles NaN correctly."""
        obj = [ClassName]()
        result = obj.[method](float('nan'))
        assert np.isnan(result) or result is not None
        
    @pytest.mark.special_values
    def test_[method]_with_infinity(self):
        """Test [method] handles infinity correctly."""
        obj = [ClassName]()
        result = obj.[method](float('inf'))
        assert result is not None
        
    # =========================================================================
    # Mock/External Dependency Tests
    # =========================================================================
    
    @pytest.mark.integration
    @patch('[dependency_path]')
    def test_[method]_with_mocked_dependency(self, mock_dep):
        """Test [method] with mocked external dependency."""
        mock_dep.return_value = expected_value
        obj = [ClassName]()
        result = obj.[method]()
        assert result == expected_value
        
    # =========================================================================
    # Edge Case Combinations
    # =========================================================================
    
    @pytest.mark.parametrize("input_val,expected_behavior", [
        ([], "empty list"),
        ({}, "empty dict"),
        ("", "empty string"),
        (0, "zero"),
        (None, "none"),
    ])
    def test_[method]_edge_cases(self, input_val, expected_behavior):
        """Test [method] with various edge case inputs."""
        obj = [ClassName]()
        # Should not raise unexpected exceptions
        try:
            result = obj.[method](input_val)
        except (TypeError, ValueError):
            pass  # Expected for invalid inputs
```

---

## Summary

This document establishes the roadmap for achieving 100% test coverage on the APGI API codebase:

1. **Current State:** ~85% core coverage, ~5% experiment coverage
2. **Gap Analysis:** ~2,626 uncovered lines across 68 experiment files and core modules
3. **Solution:** 6-week implementation plan adding ~1,500 targeted tests
4. **Outcome:** 100% line and branch coverage across all modules

**Next Action:** Begin Week 1 implementation focusing on APGI_System.py coverage gaps.
