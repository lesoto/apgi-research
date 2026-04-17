import pytest
import subprocess
import pickle
from apgi_security import validate_config_checksum


def test_subprocess_allowlist_bypass():
    with pytest.raises(PermissionError, match="not allowed"):
        subprocess.Popen(["ls", "-la"])

    with pytest.raises(PermissionError, match="not allowed"):
        subprocess.Popen("rm -rf /", shell=True)

    # Allowed command should not raise permission error
    try:
        subprocess.Popen(
            ["python", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except OSError:
        pass


def test_pickle_bypass():
    payload = b"cos\nsystem\n(S'echo hacked'\ntR."
    with pytest.raises(ValueError, match="Pickle is disabled"):
        pickle.loads(payload)


def test_config_signing_validation():
    config = {"alpha": 5.0, "beta": 1.0}
    secret = "test_key"

    import json
    import hashlib

    string_data = json.dumps(config, sort_keys=True) + secret
    valid_hash = hashlib.sha256(string_data.encode("utf-8")).hexdigest()

    # Valid
    assert validate_config_checksum(config, valid_hash, secret) is True

    # Bypass attempt (tampered value)
    tampered_config = {"alpha": 5.0, "beta": 10.0}
    assert validate_config_checksum(tampered_config, valid_hash, secret) is False
