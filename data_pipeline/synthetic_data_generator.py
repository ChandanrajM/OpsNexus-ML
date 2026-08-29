"""
Synthetic data generator for OpsNexus-ML
Generates telemetry data matching the OpsNexus MetricPayload structure
"""
import json
import random
from datetime import datetime, timedelta
import math

def generate_cpu_usage(base=30.0, daily_amplitude=20.0, weekly_amplitude=10.0, noise=5.0, timestamp=None):
    """
    Generate realistic CPU usage percentage with daily and weekly patterns
    """
    if timestamp is None:
        timestamp = datetime.now()

    # Hour of day (0-23)
    hour = timestamp.hour + timestamp.minute / 60.0
    # Day of week (0-6, Monday=0)
    day_of_week = timestamp.weekday()

    # Daily pattern: higher during day (8 AM - 6 PM), lower at night
    daily_pattern = math.sin(2 * math.pi * (hour - 8) / 12)  # Peak at 2 PM
    daily_pattern = max(0, daily_pattern)  # Only positive part

    # Weekly pattern: higher on weekdays (Mon-Fri), lower on weekend (Sat-Sun)
    weekly_pattern = 1.0 if day_of_week < 5 else 0.3  # Weekday=1.0, Weekend=0.3

    # Combine patterns
    cpu_usage = base + daily_amplitude * daily_pattern + weekly_amplitude * (weekly_pattern - 0.65) + random.gauss(0, noise)

    # Clamp between 0 and 100
    cpu_usage = max(0, min(100, cpu_usage))

    return cpu_usage

def generate_per_cpu_counts(cpu_usage, core_count=4):
    """
    Generate per-core usage percentages that average to the total cpu_usage
    """
    # Generate random variations around the average
    per_core = []
    remaining = cpu_usage

    for i in range(core_count - 1):
        # Randomly assign a portion of the remaining usage to this core
        core_usage = random.uniform(0, min(remaining * 2, 100))
        per_core.append(core_usage)
        remaining -= core_usage

    # Last core gets the remaining usage
    per_core.append(max(0, remaining))

    # Ensure we have exactly core_count elements
    while len(per_core) < core_count:
        per_core.append(0.0)

    # If we have too many, truncate
    per_core = per_core[:core_count]

    # Normalize so the average matches cpu_usage (adjust for rounding)
    current_avg = sum(per_core) / core_count
    if current_avg > 0:
        adjustment_factor = cpu_usage / current_avg
        per_core = [min(100, x * adjustment_factor) for x in per_core]

    return per_core, core_count

def generate_memory_usage(base=40.0, daily_amplitude=15.0, noise=3.0, timestamp=None):
    """
    Generate memory usage percentage
    """
    if timestamp is None:
        timestamp = datetime.now()

    hour = timestamp.hour + timestamp.minute / 60.0
    daily_pattern = math.sin(2 * math.pi * (hour - 8) / 12)  # Peak at 2 PM
    daily_pattern = max(0, daily_pattern)

    memory_usage = base + daily_amplitude * daily_pattern + random.gauss(0, noise)
    memory_usage = max(0, min(100, memory_usage))

    return memory_usage

def generate_disk_io(base_read=10.0, base_write=5.0, noise=2.0):
    """
    Generate disk I/O in MB/s (read and write)
    """
    read_mbps = max(0, base_read + random.gauss(0, noise))
    write_mbps = max(0, base_write + random.gauss(0, noise))
    return read_mbps, write_mbps

def generate_network_bytes(base_sent=1000.0, base_recv=1500.0, noise=50.0):
    """
    Generate network bytes per second
    """
    sent_bytes = max(0, base_sent + random.gauss(0, noise))
    recv_bytes = max(0, base_recv + random.gauss(0, noise))
    return sent_bytes, recv_bytes

def generate_uptime(base_hours=720.0, noise=1.0):
    """
    Generate system uptime in hours (increases over time)
    """
    return max(0, base_hours + random.gauss(0, noise))

def generate_process_count(base=50.0, daily_amplitude=20.0, noise=5.0, timestamp=None):
    """
    Generate number of processes
    """
    if timestamp is None:
        timestamp = datetime.now()

    hour = timestamp.hour + timestamp.minute / 60.0
    daily_pattern = math.sin(2 * math.pi * (hour - 8) / 12)  # Peak at 2 PM
    daily_pattern = max(0, daily_pattern)

    process_count = base + daily_amplitude * daily_pattern + random.gauss(0, noise)
    process_count = max(1, int(process_count))

    return process_count

def generate_metric_payload(agent_id="synthetic-agent-001", start_time=None, num_points=1000, interval_seconds=10):
    """
    Generate a list of metric payloads matching OpsNexus structure
    """
    if start_time is None:
        start_time = datetime.now() - timedelta(seconds=num_points * interval_seconds)

    metrics_list = []
    current_time = start_time

    for i in range(num_points):
        # Generate timestamp
        timestamp = current_time + timedelta(seconds=i * interval_seconds)
        iso_timestamp = timestamp.isoformat() + "Z"

        # Generate CPU metrics
        cpu_usage = generate_cpu_usage(timestamp=timestamp)
        per_cpu, core_count = generate_per_cpu_counts(cpu_usage)

        # Generate memory metrics
        memory_usage = generate_memory_usage(timestamp=timestamp)

        # Generate disk I/O
        disk_read, disk_write = generate_disk_io()

        # Generate network
        net_sent, net_recv = generate_network_bytes()

        # Generate uptime (accumulating)
        uptime_hours = generate_uptime(base_hours=i * interval_seconds / 3600.0)

        # Generate process count
        process_count = generate_process_count(timestamp=timestamp)

        # Build the metric payload
        payload = {
            "agent_id": agent_id,
            "timestamp": iso_timestamp,
            "metrics": {
                "system": {
                    "cpu": {
                        "usage_percent": cpu_usage,
                        "per_cpu": per_cpu,
                        "count": core_count
                    },
                    "memory": {
                        "usage_percent": memory_usage,
                        "available_mb": 8192 * (100 - memory_usage) / 100,  # Assuming 8GB total
                        "used_mb": 8192 * memory_usage / 100,
                        "free_mb": 8192 * (100 - memory_usage) / 100
                    },
                    "disk": {
                        "read_mbps": disk_read,
                        "write_mbps": disk_write,
                        "read_count": int(disk_read * 100),  # Simplified
                        "write_count": int(disk_write * 100)
                    },
                    "network": {
                        "bytes_sent": net_sent,
                        "bytes_recv": net_recv,
                        "packets_sent": int(net_sent / 100),  # Simplified
                        "packets_recv": int(net_recv / 100),
                        "errin": 0,
                        "errout": 0,
                        "dropin": 0,
                        "dropout": 0
                    },
                    "uptime": {
                        "timestamp": iso_timestamp,
                        "boot_time": (timestamp - timedelta(hours=uptime_hours)).isoformat() + "Z",
                        "uptime": uptime_hours * 3600  # Convert to seconds
                    },
                    "processes": {
                        "count": process_count,
                        "running": int(process_count * 0.8),
                        "sleeping": int(process_count * 0.15),
                        "zombie": int(process_count * 0.05),
                        "stopped": int(process_count * 0.02)
                    }
                }
            }
        }

        metrics_list.append(payload)

    return metrics_list

def save_synthetic_data(data, filename="synthetic_telemetry.json"):
    """
    Save synthetic data to a JSON file
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} data points to {filename}")

if __name__ == "__main__":
    # Generate 2 days of data at 10-second intervals
    # 2 days * 24 hours * 60 minutes * 6 intervals per hour = 17280 points
    print("Generating synthetic OpsNexus telemetry data...")
    data = generate_metric_payload(
        agent_id="synthetic-agent-001",
        num_points=17280,
        interval_seconds=10
    )

    save_synthetic_data(data, "/home/chandanraj-m/OpsNexus-ML/data_pipeline/synthetic_telemetry.json")

    # Also save a smaller sample for quick testing
    sample_data = data[:100]  # First 100 points
    save_synthetic_data(sample_data, "/home/chandanraj-m/OpsNexus-ML/data_pipeline/synthetic_telemetry_sample.json")

    print("Done!")