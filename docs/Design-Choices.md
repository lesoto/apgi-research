# Architecture Decision Record (ADR): APGI Design Choices

The APGI requires robust, scalable, and secure architectural frameworks. As the system involves running autonomous experiments and generative modeling, we need clear boundaries for compliance, security, and reproducibility.

## Decision

1. **Deny-by-default Subprocess Execution**: To minimize the risk of arbitrary code execution, subprocess calls are intercepted and filtered through an explicit allowlist.
2. **Elimination of Untrusted Pickling**: Python's `pickle` module has been explicitly blocked and overridden with secure JSON deserialization to prevent malicious payload execution during configuration loads or model state transfers.
3. **Pydantic for Data Validation**: The system uses explicitly typed schemas to manage configurations and experimental parameters, ensuring validation ahead of runtime errors.
4. **Ring-buffered Deques for History**: To prevent unbounded memory consumption across long-lived autonomous agents (long-term experiment simulation threads), fixed-size buffers (`collections.deque`) replace raw list accumulators.
5. **Compliance-by-design Boundaries**: Built-in TTLs, data classification rules, and pseudonymization steps are enforced natively before logging or caching sensitive experiment data.

## Consequences

- **Positive**: Hardened security posture, clear alignment with compliance requirements, mathematically-proven state updates, predictable memory limits.
- **Negative**: Adds overhead when extending subprocess interactions; JSON limitations force specific constraints on state serialization compared to pickle.
