# APGI Failure Triage Runbooks

## Quick Reference

| Exit Code | Meaning | Section |
| :--- | :--- | :--- |
| 0 | Success | — |
| 1 | General error | CLI Runbook Section A |
| 77 | Authorization error | CLI Runbook Section B |
| 130 | Keyboard interrupt | — |

---

## CLI Failure Triage

### General Error (Exit Code 1)

1. Check structured logs: `uv run experiments/run_*.py 2>&1`
2. Validate env vars: `env | grep APGI_`
3. Common fixes:
   - `ModuleNotFoundError`: `uv pip install -e .`
   - `AuthorizationError`: `export APGI_OPERATOR_ROLE=operator`
   - `APGI_AUDIT_KEY` missing: `export APGI_AUDIT_KEY=<64+ char key>`

### Authorization (Exit Code 77)

1. Check role: `echo $APGI_OPERATOR_ROLE` (should be operator/admin)
2. Guest role cannot run experiments by design

### Performance Issues

Enable profiling: `APGI_ENABLE_PROFILING=1 uv run experiments/run_*.py`

---

## GUI Failure Triage

### GUI Won't Start

1. Check tkinter: `python -c "import tkinter; print(tkinter.Tk)"`
2. Check customtkinter: `python -c "import customtkinter"`

### Experiment Won't Launch

1. Manual test: `uv run experiments/run_*.py --help`
2. Check subprocess: `python -c "import subprocess; print(subprocess.run(['echo', 'test'], capture_output=True))"`

### GUI Hangs

1. Check zombies: `ps aux | grep "run_" | grep -v grep`
2. Check memory: `psutil` or `vm_stat`

---

## Long-Running Pipeline Triage

### Pipeline Stops

1. Check orchestration: `apgi_orchestration_kernel.get_pipeline_status()`
2. Check resource limits: `ulimit -a`

### Memory Leaks

1. Enable tracemalloc
2. Common leaks:
   - torch tensors: `del tensor`
   - matplotlib: `plt.close('all')`
   - zombies: always call `communicate()`

### Recovery

Resume from checkpoint:

```python
from apgi_orchestration_kernel import CheckpointManager
checkpoint_mgr.load_latest()
```

---

## Full Documentation

See detailed runbooks in docs/:

- Entry point contract: `docs/adr/001-entry-point-contract.md`
- Security model: `docs/adr/002-security-model.md`
- APGI architecture: `docs/adr/003-apgi-engine-architecture.md`
