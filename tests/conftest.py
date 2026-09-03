"""
Pytest configuration and shared fixtures
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np


@pytest.fixture(scope="session")
def random_seed():
    """Set random seed for reproducible tests"""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_telemetry_data():
    """Generate sample telemetry data for testing"""
    np.random.seed(42)
    n_samples = 100
    data = []

    for i in range(n_samples):
        record = {
            "agent_id": f"agent-{np.random.randint(1, 5)}",
            "timestamp": 1700000000 + i * 60,
            "cpu_usage_percent": max(0, min(100, np.random.normal(30, 15))),
            "memory_usage_percent": max(0, min(100, np.random.normal(45, 20))),
            "disk_usage_percent": max(0, min(100, np.random.normal(60, 10))),
            "network_bytes_sent": max(0, np.random.exponential(1000000)),
            "network_bytes_recv": max(0, np.random.exponential(800000)),
            "disk_read_bytes": max(0, np.random.exponential(500000)),
            "disk_write_bytes": max(0, np.random.exponential(300000)),
            "process_count": np.random.randint(50, 300),
            "thread_count": np.random.randint(100, 1000),
            "load_average_1m": max(0, np.random.normal(1.5, 1.0)),
            "load_average_5m": max(0, np.random.normal(1.3, 0.8)),
            "load_average_15m": max(0, np.random.normal(1.2, 0.6)),
            "uptime_seconds": 1700000000 + i * 60
        }
        data.append(record)

    return data


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for tests"""
    return tmp_path


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "api: mark test as API test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on path"""
    for item in items:
        # Add markers based on test location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "api" in str(item.fspath):
            item.add_marker(pytest.mark.api)