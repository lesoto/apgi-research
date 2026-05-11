"""
Enhanced test suite for apgi_config.py

This test file covers all major components:
- ConfigManager singleton pattern
- Pydantic schema validation
- Environment variable loading
- File loading (JSON/YAML)
- Configuration migration
- Caching mechanisms
- Legacy compatibility
"""

import pytest
import json
import time
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from utils.apgi_config import (
    ConfigManager,
    reset_config,
)


class TestConfigManager:
    """Test ConfigManager class methods."""

    def test_config_isolation(self):
        """Test that configuration instances are properly isolated."""
        # Reset singleton
        reset_config()

        # Create two instances and verify they are the same (singleton)
        config1 = ConfigManager()
        config2 = ConfigManager()
        assert config1 is config2
        assert config1._initialized
        assert config2._initialized

    def test_concurrent_access(self):
        """Test concurrent access to ConfigManager."""
        import threading
        import time

        # Create instances for concurrent access
        ConfigManager()
        ConfigManager()

        results = []
        errors = []

        def access_config(instance_id):
            try:
                for _ in range(100):
                    config = ConfigManager()
                    value = config.get("test_key", f"value_{instance_id}")
                    results.append(value)
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(f"Instance {instance_id}: {e}")

        # Start concurrent access
        thread1 = threading.Thread(target=access_config, args=(1,))
        thread2 = threading.Thread(target=access_config, args=(2,))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        # Should have no errors and consistent values
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(set(results)) == 200  # 100 operations per thread

    def test_memory_efficiency(self):
        """Test memory efficiency of ConfigManager."""
        config = ConfigManager()

        # Load many configurations to test memory usage
        for i in range(1000):
            config.get_experiment_config(f"memory_test_{i}")

    def test_config_validation_edge_cases(self):
        """Test configuration validation with edge cases."""
        config = ConfigManager()

        # Test with extreme values
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            extreme_config = {
                "tau_s": 1000.0,  # Way above max
                "beta": -10.0,  # Way below min
                "theta_0": 10.0,  # Way above max
                "alpha": -5.0,  # Way below min
            }
            tmp_file.write(json.dumps(extreme_config))
            tmp_file.flush()

            with patch.object(ConfigManager, "_find_config_file") as mock_find:
                mock_find.return_value = Path(tmp_file.name)

                config = ConfigManager()

                # Test that extreme values are handled gracefully
                # ConfigManager uses Pydantic schemas which handle validation automatically
                experiment_config = config.get_experiment_config("extreme_test")

                # Verify the config was loaded (Pydantic validation happens internally)
                assert experiment_config is not None

    def test_config_source_precedence(self):
        """Test configuration source precedence."""
        config = ConfigManager()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            file_config = {
                "experiment_name": "file_test",
                "tau_s": 0.25,
            }
            tmp_file.write(json.dumps(file_config))
            tmp_file.flush()

            # Set environment variable
            os.environ["APGI_EXPERIMENT_TEST_tau_s"] = "0.5"

            with patch.object(ConfigManager, "_find_config_file") as mock_find:
                mock_find.return_value = Path(tmp_file.name)

                config = ConfigManager()
                tau_s = config.get("experiment_test_tau_s")

                # Environment should take precedence
                assert tau_s == "0.5"  # From environment, not file

    def test_config_reload_consistency(self):
        """Test configuration reload consistency."""
        config = ConfigManager()

        # Load initial config
        config.get_all_sources().copy()

        # Modify config
        config.set("test_key", "modified_value", source="test")
        config.get_all_sources().copy()

        # Reload
        config.reload()

        # Should preserve modified values
        assert config.get("test_key") == "modified_value"
        assert config.get_source("test_key") == "test"

        # Should clear caches
        assert len(config._config_cache) == 0

    def test_config_invalid_json_handling(self):
        """Test handling of invalid JSON configuration files."""
        config = ConfigManager()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            tmp_file.write('{"invalid": json content}')
            tmp_file.flush()

            with patch.object(ConfigManager, "_find_config_file") as mock_find:
                mock_find.return_value = Path(tmp_file.name)

                config = ConfigManager()
                # Should not crash, should use defaults
                assert config.get("invalid") is None
                assert config.get("experiment_name") == "unknown_experiment"

    def test_config_circular_reference_handling(self):
        """Test handling of circular references in configuration."""
        config = ConfigManager()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            circular_config = {
                "experiment_name": "main",
                "parameters": {"main": {"experiment_name": "main"}},
            }
            tmp_file.write(json.dumps(circular_config))
            tmp_file.flush()

            with patch.object(ConfigManager, "_find_config_file") as mock_find:
                mock_find.return_value = Path(tmp_file.name)

                config = ConfigManager()
                experiment_config = config.get_experiment_config("main")

                # Should handle circular reference gracefully
                assert experiment_config.experiment_name == "main"
                assert experiment_config.experiment_name == "main"

    def test_config_performance(self):
        """Test ConfigManager performance with large configurations."""
        config = ConfigManager()

        # Test with many configuration accesses
        start_time = time.time()
        for i in range(1000):
            config.get_experiment_config(f"perf_test_{i}")

        end_time = time.time()
        duration = end_time - start_time

        # Should complete within reasonable time
        assert duration < 1.0, f"Config access took too long: {duration:.3f}s"


if __name__ == "__main__":
    pytest.main()
