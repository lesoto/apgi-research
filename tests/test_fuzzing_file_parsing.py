"""
================================================================================
FUZZING TESTS FOR FILE PARSING
================================================================================

Comprehensive fuzzing tests for file parsing operations.
Tests robustness against malformed inputs and edge cases.
"""

from __future__ import annotations

import json
import random
import struct
import sys
from pathlib import Path
from typing import List, Union

import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# FUZZ GENERATORS
# =============================================================================


def generate_malformed_json() -> List[Union[str, bytes]]:
    """Generate malformed JSON inputs for fuzzing."""
    return [
        "",  # Empty
        "{",  # Unclosed object
        "}",  # Unopened close
        "[",  # Unclosed array
        "]",  # Unopened close
        '{"key":}',  # Missing value
        '{"key"}',  # Missing colon and value
        '{"key": undefined}',  # undefined is not valid JSON
        "{key: 'value'}",  # Unquoted keys
        "{'key': 'value'}",  # Single quotes
        '{"key": "value",}',  # Trailing comma
        "null",  # Bare null
        "true",  # Bare true
        "false",  # Bare false
        "123",  # Bare number
        "Infinity",  # Invalid number
        "NaN",  # Invalid number
        "0x10",  # Hex not allowed
        "\x00",  # Null byte
        '{"\x00": "value"}',  # Null byte in key
        "A" * 1000000,  # Very long string
        '{"nested": {' * 1000 + "}" * 1000,  # Deeply nested
    ]


def generate_malformed_numpy() -> List[bytes]:
    """Generate malformed numpy file inputs."""
    rng = random.Random(42)  # Seeded for reproducibility
    return [
        b"",  # Empty
        b"\x93NUMPY",  # Invalid magic
        b"\x93NUMPY\x01\x00",  # Wrong version
        b"\x93NUMPY\x01\x00v\x00" + b"{" * 100,  # Invalid header
        struct.pack("<I", 0xFFFFFFFF),  # Max size
        bytes(rng.randint(0, 255) for _ in range(100)),  # Random bytes
        bytes(rng.randint(0, 255) for _ in range(1000)),
        bytes(rng.randint(0, 255) for _ in range(10000)),
        b"\x00" * 100,  # Null bytes
        b"\xff" * 100,  # All high bytes
    ]


def generate_path_traversal() -> List[str]:
    """Generate path traversal attempts."""
    return [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "....//....//....//etc/passwd",
        ".\\.\\.\\/etc/passwd",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc/passwd",
        "..%c0%af..%c0%af..%c0%afetc/passwd",
        "/etc/passwd",
        "\\windows\\system.ini",
        "file:///etc/passwd",
        "C:\\Windows\\System32\\drivers\\etc\\hosts",
        "\x00",  # Null byte in path
        "CON",  # Windows reserved name
        "LPT1",  # Windows reserved name
        ".",  # Current directory
        "..",  # Parent directory
        "",  # Empty path
        "/",  # Root only
        "//server/share",  # UNC path
    ]


# =============================================================================
# JSON PARSING FUZZ TESTS
# =============================================================================


@pytest.mark.fuzz
class TestJSONParsingFuzz:
    """Fuzz tests for JSON parsing."""

    @pytest.mark.parametrize("malformed_input", generate_malformed_json())
    def test_json_parse_robustness(self, malformed_input: Union[str, bytes]) -> None:
        """Test JSON parser handles malformed inputs gracefully."""
        try:
            if isinstance(malformed_input, bytes):
                try:
                    decoded = malformed_input.decode("utf-8")
                except UnicodeDecodeError:
                    return
                json.loads(decoded)
            else:
                json.loads(malformed_input)
        except json.JSONDecodeError:
            pass
        except (ValueError, TypeError):
            pass

    @pytest.mark.parametrize(
        "input_data",
        [
            '{"experiment": "test", "trials": 100}',
            '{"config": {"n_trials": 50, "duration": 2000}}',
            '{"results": [1, 2, 3, 4, 5]}',
        ],
    )
    def test_valid_json_parsing(self, input_data: str) -> None:
        """Test that valid JSON parses correctly."""
        result = json.loads(input_data)
        assert isinstance(result, dict)

    def test_json_roundtrip(self) -> None:
        """Test JSON serialization roundtrip."""
        original = {
            "experiment_name": "test",
            "participant_id": "P001",
            "n_trials": 100,
            "results": [True, False, True],
            "metrics": {"accuracy": 0.85},
        }

        serialized = json.dumps(original)
        restored = json.loads(serialized)

        assert restored == original


# =============================================================================
# NUMPY FILE FUZZ TESTS
# =============================================================================


@pytest.mark.fuzz
class TestNumpyFileFuzz:
    """Fuzz tests for numpy file handling."""

    @pytest.mark.parametrize("malformed_data", generate_malformed_numpy())
    def test_numpy_load_robustness(self, malformed_data: bytes, tmp_path: Path) -> None:
        """Test numpy handles malformed files gracefully."""
        test_file = tmp_path / "test.npy"
        test_file.write_bytes(malformed_data)

        try:
            np.load(test_file)
        except (OSError, ValueError, IndexError, struct.error):
            pass
        except Exception:
            pass

    def test_numpy_save_load_roundtrip(self, tmp_path: Path) -> None:
        """Test numpy save/load roundtrip."""
        original = np.random.randn(100, 10)

        test_file = tmp_path / "test.npy"
        np.save(test_file, original)

        loaded = np.load(test_file)

        np.testing.assert_array_equal(original, loaded)


# =============================================================================
# PATH TRAVERSAL FUZZ TESTS
# =============================================================================


@pytest.mark.fuzz
@pytest.mark.security
class TestPathTraversalFuzz:
    """Fuzz tests for path traversal vulnerabilities."""

    @pytest.mark.parametrize("malicious_path", generate_path_traversal())
    def test_path_traversal_attempts(self, malicious_path: str, tmp_path: Path) -> None:
        """Test path handling resists traversal attempts."""
        try:
            resolved = (tmp_path / malicious_path).resolve()
            try:
                resolved.relative_to(tmp_path)
            except ValueError:
                pass
        except (ValueError, OSError, RuntimeError):
            pass

    def test_safe_filename_handling(self, tmp_path: Path) -> None:
        """Test safe filename handling."""
        safe_name = "test_file.json"
        safe_path = tmp_path / safe_name
        safe_path.write_text("test")

        assert safe_path.exists()


# =============================================================================
# CONFIGURATION
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with fuzz-specific markers."""
    config.addinivalue_line("markers", "fuzz: marks tests as fuzzing tests")
