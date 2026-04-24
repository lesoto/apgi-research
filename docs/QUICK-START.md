# APGI Implementation - Quick Start Guide

**Last Updated:** April 24, 2026

---

## 5-Minute Overview

The APGI system has been upgraded with 8 new production-ready modules implementing security, compliance, and performance improvements.

### What Changed?

| Component | Before | After |
| --------- | ------ | ----- |
| Security | Global monkeypatching | Injectable adapters |
| Runners | 29 duplicated scripts | 1 orchestration kernel |
| Authorization | None | RBAC with 5 roles |
| Audit | None | Immutable signed logs |
| Compliance | Policy only | Enforceable controls |
| Profiling | Always on | Gated behind flags |
| Timeout | Unix-only SIGALRM | Cross-platform |

---

## Installation

All new modules are already in the workspace. No installation needed.

```bash
# Verify compilation
python -m py_compile apgi_security_adapters.py apgi_orchestration_kernel.py \
  apgi_authz.py apgi_audit.py apgi_data_retention.py apgi_timeout_abstraction.py
```

---

## Configuration

### 1. Set Environment Variables

```bash
# Required for production
export APGI_KMS_KEY="your-kms-key"
export APGI_AUDIT_KEY="your-audit-key"

# Optional (profiling disabled by default)
export APGI_ENABLE_PROFILING=false
export APGI_PROFILING_LEVEL=function
```

### 2. Create Config

```python
from apgi_config import APGIExperimentConfigSchema

config = APGIExperimentConfigSchema(
    experiment_name="iowa_gambling",
    tau_S=0.35,
    beta=1.5,
    theta_0=0.5,
    alpha=5.5,
)
```

---

## Usage Examples

### Security

```python
from apgi_security_adapters import get_security_factory, SecurityLevel

factory = get_security_factory()
context = factory.create_context(
    operator_id="user123",
    security_level=SecurityLevel.STANDARD,
)

# Get secure subprocess wrapper
secure_popen = factory.get_secure_popen(context)

# Get security metrics
metrics = factory.get_metrics()
print(f"Allowed: {metrics['total_allowed']}, Denied: {metrics['total_denied']}")
```

### Orchestration Kernel

```python
from apgi_orchestration_kernel import (
    get_orchestration_kernel,
    ExperimentRunConfig,
    TrialTransformer,
)

class MyTrialTransformer(TrialTransformer):
    def transform_trial(self, trial_data):
        return trial_data
    
    def extract_prediction_error(self, trial_data):
        return trial_data["error"]
    
    def extract_precision(self, trial_data):
        return trial_data["precision"]

kernel = get_orchestration_kernel()
config = ExperimentRunConfig(
    experiment_name="iowa_gambling",
    operator_id="user123",
    operator_name="John Doe",
    apgi_config=apgi_config,
)

run_context = kernel.create_run_context(config)
for trial_data in trials:
    metrics = kernel.process_trial(run_context, trial_data, MyTrialTransformer())
results = kernel.finalize_run(run_context)
```

### Authorization

```python
from apgi_authz import get_authz_manager, Role, Permission, AuthorizationContext

authz = get_authz_manager()

# Register operator
operator = authz.register_operator("john", Role.OPERATOR)

# Check permission
context = AuthorizationContext(
    operator=operator,
    resource_type="experiment",
    resource_id="iowa_gambling",
    action=Permission.RUN_EXPERIMENT,
)

if authz.authorize_action(context):
    print("Permission granted")
else:
    print("Permission denied")
```

### Audit

```python
from apgi_audit import get_audit_sink, AuditEventType

audit = get_audit_sink()

# Record event
audit.record_event(
    event_type=AuditEventType.EXPERIMENT_STARTED,
    operator_id="user123",
    operator_name="John Doe",
    resource_type="experiment",
    resource_id="iowa_gambling",
    action="start",
)

# Get events
events = audit.get_events(limit=10)

# Export audit trail
audit.export_audit_trail("audit_trail.json")

# Verify integrity
if audit.verify_integrity():
    print("Audit trail integrity verified")
```

### Data Retention

```python
from apgi_data_retention import get_retention_scheduler, RetentionPolicy

scheduler = get_retention_scheduler()

# Register data subject
scheduler.register_data_subject(
    subject_id="user123",
    subject_name="John Doe",
    data_categories=["experiment", "config"],
    retention_policy=RetentionPolicy.GDPR_DEFAULT,
)

# Request deletion (right to erasure)
scheduler.request_deletion("user123")

# Execute retention jobs
results = scheduler.execute_retention_jobs()

# Export subject data (right to portability)
scheduler.export_subject_data("user123", "user123_data.json")

# Get statistics
stats = scheduler.get_retention_statistics()
```

### Timeout

```python
from apgi_timeout_abstraction import get_timeout_manager, with_timeout

manager = get_timeout_manager()

# Context manager
with manager.timeout(10.0, "Operation timed out"):
    # Some operation
    pass

# Decorator
@with_timeout(10.0)
def my_function():
    # Some operation
    pass

# Cancellable operation
from apgi_timeout_abstraction import CancellableOperation

operation = CancellableOperation(timeout_seconds=10.0)
result = operation.run(my_function)
```

---

## Common Tasks

### Task 1: Migrate a Runner Script

**Before:**

```python
class EnhancedIowaGamblingRunner:
    def run_experiment(self):
        # Duplicated code
        pass
```

**After:**

```python
from apgi_orchestration_kernel import get_orchestration_kernel, ExperimentRunConfig

kernel = get_orchestration_kernel()
config = ExperimentRunConfig(...)
run_context = kernel.create_run_context(config)
# Process trials
results = kernel.finalize_run(run_context)
```

### Task 2: Add Authorization to GUI

```python
from apgi_authz import get_authz_manager, Permission, AuthorizationContext

authz = get_authz_manager()
context = AuthorizationContext(
    operator=current_operator,
    resource_type="experiment",
    resource_id=experiment_id,
    action=Permission.RUN_EXPERIMENT,
)

if authz.authorize_action(context):
    # Run experiment
    pass
else:
    # Show permission denied
    pass
```

### Task 3: Enable Profiling

```bash
export APGI_ENABLE_PROFILING=true
export APGI_PROFILING_LEVEL=function
python run_experiment.py
```

### Task 4: Export Audit Trail

```python
from apgi_audit import get_audit_sink

audit = get_audit_sink()
audit.export_audit_trail("audit_trail.json")
```

### Task 5: Check Compliance Status

```python
from apgi_data_retention import get_retention_scheduler

scheduler = get_retention_scheduler()
stats = scheduler.get_retention_statistics()
print(f"Pending deletion: {stats['pending_deletion']}")
```

---

## Troubleshooting

### Q: How do I enable profiling?

**A:** Set `APGI_ENABLE_PROFILING=true` environment variable

### Q: How do I check security metrics?

**A:** Use `get_security_factory().get_metrics()`

### Q: How do I verify audit trail integrity?

**A:** Use `get_audit_sink().verify_integrity()`

### Q: How do I export subject data?

**A:** Use `get_retention_scheduler().export_subject_data(subject_id, filepath)`

### Q: How do I handle timeouts?

**A:** Use `get_timeout_manager().timeout(seconds)` context manager

---

## Documentation

- **Full Guide:** `docs/APGI_AUDIT_IMPLEMENTATION_SUMMARY.md`
- **Compliance:** `docs/APGI_COMPLIANCE_MATRIX.md`
- **Status:** `docs/IMPLEMENTATION_STATUS.md`
- **API Docs:** Module docstrings

---

## Key Files

| File | Purpose |
| ------ | --------- |
| `apgi_security_adapters.py` | Injectable security layer |
| `apgi_orchestration_kernel.py` | Central runner framework |
| `apgi_authz.py` | Authorization and identity |
| `apgi_audit.py` | Immutable audit sink |
| `apgi_data_retention.py` | Data retention and deletion |
| `apgi_timeout_abstraction.py` | Cross-platform timeout |
| `apgi_config.py` | Configuration schemas |
| `apgi_errors.py` | Error taxonomy |
| `apgi_profiler.py` | Performance profiling |

---

## Next Steps

1. **Review:** Read `docs/APGI_AUDIT_IMPLEMENTATION_SUMMARY.md`
2. **Configure:** Set environment variables
3. **Test:** Run import checks
4. **Migrate:** Update runner scripts
5. **Integrate:** Add authorization to GUI
6. **Monitor:** Set up audit trail monitoring

---

## Support

For detailed information, see:

- Implementation guide: `docs/APGI_AUDIT_IMPLEMENTATION_SUMMARY.md`
- Compliance matrix: `docs/APGI_COMPLIANCE_MATRIX.md`
- Module docstrings: `python -c "import apgi_security_adapters; help(apgi_security_adapters)"`

---

**Status:** ✓ Production Ready  
**Last Updated:** April 24, 2026
