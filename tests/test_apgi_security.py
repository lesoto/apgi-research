import pytest

from apgi_security import (
    PickleSecurityError,
    SecureSubprocessError,
    secure_loads,
    secure_popen,
    validate_config_checksum,
)


def test_secure_popen_blocks_disallowed_commands():
    """Test that secure_popen blocks commands not in allowlist."""
    with pytest.raises(SecureSubprocessError, match="not in the allowlist"):
        secure_popen(["rm", "-rf", "/"])

    with pytest.raises(SecureSubprocessError, match="not in the allowlist"):
        secure_popen("ls -la", shell=True)


def test_secure_loads_blocks_pickle_by_default():
    """Test that secure_loads blocks pickle by default."""
    payload = b"cos\nsystem\n(S'echo hacked'\ntR."
    with pytest.raises(PickleSecurityError, match="Pickle is disabled"):
        secure_loads(payload)


def test_config_signing_validation():
    config = {"alpha": 5.0, "beta": 1.0}
    secret = "test_key"

    import hashlib
    import json

    string_data = json.dumps(config, sort_keys=True) + secret
    valid_hash = hashlib.sha256(string_data.encode("utf-8")).hexdigest()

    # Valid
    assert validate_config_checksum(config, valid_hash, secret) is True

    # Bypass attempt (tampered value)
    tampered_config = {"alpha": 5.0, "beta": 10.0}
    assert validate_config_checksum(tampered_config, valid_hash, secret) is False
