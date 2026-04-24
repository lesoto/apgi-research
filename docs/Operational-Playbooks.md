# APGI Operational Playbooks

## 1. Incident Response Playbook

**Scope:** Responding to APGI system crashes, LLM integration failures, or security boundary violations.

**Steps:**

1. **Triage**: Check `apgi_logging` outputs to identify the faulting module (e.g., Engine, Validator, Subprocess wrapper).
2. **Containment**: If an untrusted execution boundary is breached, halt the Celery worker queue immediately and lock the environment.
3. **Investigation**: Query the `ComplianceManager` audit trails to identify the latest parameter changes or experiment states leading to the crash.
4. **Resolution**: Roll back `config/default.yaml` to the last known robust state and restart the main agent loop.

## 2. Timeout Tuning Playbook

**Scope:** Handling `TimeoutError` in asynchronous agents and long-running batch processing.

**Steps:**

1. **Metrics Review**: Inspect `avg_execution_time` and `volatility` from `XPRAgentEngineEnhanced.analyze_performance_trend()`.
2. **Adjustments**: If baseline latency has organically grown, increase bounds in configuration; else check for run-away recursive loops or HTTP blockages on the LLM API.
3. **Validation**: Test temporal parameters in isolation using `test_system_integration.py`.

## 3. Performance Regressions Playbook

**Scope:** Identifying drops in evaluation matrices or spikes in memory allocation.

**Steps:**

1. **Profiling**: Run the `apgi_profiler.py` and inspect matrix multiplication hotspots in `process_trials`.
2. **Memory Leaks**: Ensure all new history logs and metric accumulators utilize a bounded structure.
3. **Re-baseline**: If algorithm optimizations inherently changed execution complexity, document the new baseline overhead in the tracking benchmarks.
