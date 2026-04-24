# APGI Experiment Runners - Maturity Matrix

| Experiment Runner | Data Classification Support | Audit Trails | Parameter Pseudonymization | Current Maturity Level | Target Readiness |
| ------------------- | ----------------------------- | -------------- | ---------------------------- | ------------------------ | ------------------ |
| `base_experiment` | Yes | Yes | Yes | Level 4: Production | Level 5: Optimized |
| `run_ai_benchmarking` | Yes | Yes | No | Level 3: Hardened | Level 4 |
| `run_attentional_blink` | Default (Internal) | Basic | Yes | Level 3: Hardened | Level 4 |
| `run_binocular_rivalry` | Default (Internal) | Basic | Partial | Level 2: Baseline | Level 3 |
| `run_change_blindness` | Default (Internal) | Basic | Partial | Level 2: Baseline | Level 3 |
| `run_drm_false_memory` | Yes | Yes | Yes | Level 4: Production | Level 4 |
| `run_dual_n_back` | Yes | Yes | Yes | Level 4: Production | Level 4 |
| `run_stroop_effect` | Default (Internal) | Partial | No | Level 2: Baseline | Level 4 |
| `run_visual_search` | Default (Internal) | Partial | No | Level 1: MVP | Level 3 |
| `run_working_memory` | Yes | Yes | Yes | Level 4: Production | Level 4 |

**Maturity Level Definitions:**

- **Level 1 (MVP)**: Runs successfully but lacks performance tuning, compliance controls, and robust error recovery.
- **Level 2 (Baseline)**: Includes basic guardrails, test coverage, and bounded memory footprints.
- **Level 3 (Hardened)**: Full security integration (no pickle, subprocess denial), config validation, and CI/CD ready.
- **Level 4 (Production)**: Compliant (TTL, classification, audit), high throughput batch processing, fully vectorized.
- **Level 5 (Optimized)**: Adaptive self-healing, LLM loop tuning, minimal latency.
