# APGI Experiment Protocol Verification Report

## Summary

- Total experiments: 28
- Complete protocols: 28/28 (100.0%) ✅
- Incomplete protocols: 0 ✅

## Verification Criteria Passed

All 28 experiments pass all 6 verification criteria:

1. **File Structure**: Both `prepare_*.py` and `run_*.py` files exist
2. **READ-ONLY Designation**: All prepare files properly marked as READ-ONLY
3. **AGENT-EDITABLE Designation**: All run files properly marked as AGENT-EDITABLE
4. **Time Budget**: All use 600-second TIME_BUDGET
5. **Primary Metrics**: All define and output primary metrics correctly
6. **Import Structure**: All run files properly import from their prepare files

## Complete Protocol List

All experiments now fully compliant:

- ai_benchmarking
- artificial_grammar_learning
- attentional_blink
- binocular_rivalry
- change_blindness
- drm_false_memory
- dual_n_back
- eriksen_flanker
- go_no_go
- inattentional_blindness
- interoceptive_gating
- iowa_gambling_task
- masking
- metabolic_cost
- multisensory_integration
- navon_task
- posner_cueing
- probabilistic_category_learning
- serial_reaction_time
- simon_effect
- somatic_marker_priming
- sternberg_memory
- stop_signal
- stroop_effect
- time_estimation
- virtual_navigation
- visual_search
- working_memory_span

## Auto-Improvement System Ready

All protocols are now fully ready for the autonomous optimization system:

- Proper file structure with clear separation of concerns
- Fixed configurations in READ-ONLY prepare files
- Agent-modifiable parameters in AGENT-EDITABLE run files
- Consistent time budgeting (600 seconds per experiment)
- Standardized primary metric output for automated parsing
- Proper import dependencies and data structures

### Template for READ-ONLY Markers

```python
"""
Fixed constants and data preparation for [Experiment] experiments.

This file is READ-ONLY. Do not modify.
It defines the fixed [configurations], task parameters, and evaluation metrics.

Usage:
    python prepare_[experiment].py  # Verify [experiment] configurations

[Experiment] paradigmsms:
[Brief description of paradigm]
"""
```
