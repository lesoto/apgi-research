# APGI Compliance Control Matrix

## Overview

This document maps APGI system controls to regulatory requirements under GDPR, CCPA, and HIPAA.

## Regulatory Framework

### GDPR (General Data Protection Regulation)

- **Scope:** EU residents' personal data
- **Key Articles:** 5 (principles), 6 (lawfulness), 17 (right to erasure), 32 (security)

### CCPA (California Consumer Privacy Act)

- **Scope:** California residents' personal information
- **Key Rights:** Access, deletion, opt-out, non-discrimination

### HIPAA (Health Insurance Portability and Accountability Act)

- **Scope:** Protected Health Information (PHI)
- **Key Rules:** Privacy Rule, Security Rule, Breach Notification Rule

---

## Control Matrix

### 1. Data Subject Rights

| Control | GDPR Article | CCPA Section | HIPAA Rule | Implementation |
| :--- | :--- | :--- | :--- | :--- |
| **Right to Access** | 15 | 1798.100 | 164.524 | `apgi_audit.py` : Export audit trail; `apgi_data_retention.py` : Subject data export |
| **Right to Erasure** | 17 | 1798.105 | 164.524(b) | `apgi_data_retention.py` : Deletion executor; Immutable audit trail preserved |
| **Right to Rectification** | 16 | N/A | 164.526 | Config validation in `apgi_config.py` ; Audit trail in `apgi_audit.py` |
| **Right to Portability** | 20 | 1798.100(d) | N/A | `apgi_data_retention.py` : Export in standard formats (JSON, CSV) |
| **Right to Object** | 21 | 1798.120 | N/A | `apgi_authz.py` : Operator consent tracking |

### 2. Data Protection & Security

| Control | GDPR Article | CCPA Section | HIPAA Rule | Implementation |
| :--- | :--- | :--- | :--- | :--- |
| **Encryption at Rest** | 32(1)(b) | 1798.150 | 164.312(a)(2)(ii) | `apgi_security_adapters.py` : Serialization format controls |
| **Encryption in Transit** | 32(1)(b) | 1798.150 | 164.312(c)(1) | TLS enforcement (external) |
| **Access Control** | 32(1)(b) | 1798.100 | 164.312(a)(2)(i) | `apgi_authz.py` : RBAC with role-based permissions |
| **Audit Logging** | 32(1)(g) | 1798.100 | 164.312(b) | `apgi_audit.py` : Immutable audit sink with integrity chain |
| **Pseudonymization** | 32(1)(a) | 1798.100 | 164.502(b) | `apgi_security_adapters.py` : Operator ID masking |
| **Integrity Verification** | 32(1)(b) | 1798.150 | 164.312(c)(2) | `apgi_audit.py` : HMAC signatures and hash chain verification |

### 3. Data Retention & Deletion

| Control | GDPR Article | CCPA Section | HIPAA Rule | Implementation |
| :--- | :--- | :--- | :--- | :--- |
| **Retention Limits** | 5(1)(e) | 1798.105 | 164.504(b) | `apgi_data_retention.py` : Configurable retention policies |
| **Deletion Execution** | 17 | 1798.105 | 164.504(b) | `apgi_data_retention.py` : Real deletion executors (not simulated) |
| **Deletion Verification** | 17 | 1798.105 | 164.504(b) | `apgi_data_retention.py` : Deletion audit trail |
| **Key Destruction** | 32(1)(b) | 1798.150 | 164.312(a)(2)(ii) | `apgi_data_retention.py` : KMS key destruction workflows |

### 4. Operator & Consent Management

| Control | GDPR Article | CCPA Section | HIPAA Rule | Implementation |
| :--- | :--- | :--- | :--- | :--- |
| **Operator Identity** | 32(1)(b) | 1798.100 | 164.308(a)(3)(ii) | `apgi_authz.py` : OperatorIdentity with audit trail |
| **Consent Tracking** | 7 | 1798.100 | 164.508 | `apgi_audit.py` : Consent events in immutable log |
| **Role-Based Access** | 32(1)(b) | 1798.100 | 164.308(a)(4) | `apgi_authz.py` : Role enum with permission mapping |
| **Audit Provenance** | 32(1)(g) | 1798.100 | 164.312(b) | `apgi_audit.py` : Signed action logs with operator tracking |

### 5. Data Processing & Transfers

| Control | GDPR Article | CCPA Section | HIPAA Rule | Implementation |
| :--- | :--- | :--- | :--- | :--- |
| **Data Processing Agreement** | 28 | 1798.100 | 164.504(e) | Documentation in `docs/` |
| **Subprocessor Management** | 28(2) | 1798.100 | 164.504(e) | `apgi_security_adapters.py` : Subprocess allowlist |
| **Data Transfer Controls** | 44-49 | 1798.100 | 164.504(e) | `apgi_security_adapters.py` : Serialization format controls |

### 6. Breach & Incident Response

| Control | GDPR Article | CCPA Section | HIPAA Rule | Implementation |
| :--- | :--- | :--- | :--- | :--- |
| **Breach Detection** | 33 | 1798.82 | 164.400 | `apgi_audit.py` : Security event logging |
| **Breach Notification** | 33-34 | 1798.82 | 164.404 | `apgi_audit.py` : Audit trail export for incident response |
| **Incident Logging** | 32(1)(g) | 1798.100 | 164.312(b) | `apgi_audit.py` : Immutable incident log |

---

## Implementation Status

### Completed (✓)

- ✓ Authorization framework (`apgi_authz.py`)
- ✓ Audit sink with integrity (`apgi_audit.py`)
- ✓ Security adapters (`apgi_security_adapters.py`)
- ✓ Error taxonomy (`apgi_errors.py`)
- ✓ Config schema with security/performance (`apgi_config.py`)

### In Progress (⏳)

- ⏳ Data retention policies (`apgi_data_retention.py`)
- ⏳ Deletion executors
- ⏳ KMS key destruction workflows
- ⏳ Consent tracking integration

### Pending (⏸)

- ⏸ Data Processing Agreement (DPA) template
- ⏸ Subprocessor list and agreements
- ⏸ Breach notification procedures
- ⏸ Data subject request workflows

---

## Key Principles

### 1. Privacy by Design

- Operator identity tracking from experiment start
- Immutable audit trail for all operations
- Role-based access control with least privilege

### 2. Data Minimization

- Only collect APGI-relevant trial metrics
- Configurable retention policies
- Automatic deletion after retention period

### 3. Transparency

- Audit trail export for data subjects
- Clear operator identity in all logs
- Documented control matrix (this document)

### 4. Accountability

- Signed action logs with operator provenance
- Integrity verification of audit trail
- Audit event export for compliance reviews

---

## Compliance Checklist

### GDPR

- [ ] Data Processing Agreement in place
- [ ] Lawful basis documented (Article 6)
- [ ] Data subject rights implemented (Articles 15-22)
- [ ] Security measures in place (Article 32)
- [ ] Audit trail maintained (Article 32(1)(g))
- [ ] Breach notification procedure (Article 33)

### CCPA

- [ ] Privacy policy updated
- [ ] Consumer rights implemented (1798.100-1798.120)
- [ ] Opt-out mechanism (1798.120)
- [ ] Non-discrimination policy (1798.125)
- [ ] Deletion verification (1798.105)

### HIPAA

- [ ] Business Associate Agreement (BAA) in place
- [ ] Access controls implemented (164.312(a)(2)(i))
- [ ] Audit controls in place (164.312(b))
- [ ] Encryption standards met (164.312(a)(2)(ii))
- [ ] Breach notification procedure (164.400-404)

---

## References

- [GDPR Official Text](https://gdpr-info.eu/)
- [CCPA Official Text](https://oag.ca.gov/privacy/ccpa)
- [HIPAA Official Text](https://www.hhs.gov/hipaa/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
