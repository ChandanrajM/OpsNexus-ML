#!/usr/bin/env python3
"""
Real-time Dashboard for OpsNexus-ML
Displays live system metrics, model predictions, and anomaly detection status
"""

import sys
import os
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
import numpy as np

# Add project directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_pipeline'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from datetime import datetime
    import time
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Dashboard will run in text-only mode.")

class OpsNexusMLDashboard:
    """Real-time dashboard for OpsNexus-ML system"""

    def __init__(self, api_base_url: str = "http://localhost:5000", refresh_interval: int = 5):
        """
        Initialize the dashboard

        Args:
            api_base_url: Base URL for the OpsNexus-ML API
            refresh_interval: Seconds between data refreshes
        """
        self.api_base_url = api_base_url
        self.refresh_interval = refresh_interval
        self.running = False

        # Data storage for trending
        self.cpu_history = []
        self.prediction_history = []
        self.timestamp_history = []
        self.anomaly_history = []
        self.max_history_points = 50  # Keep last 50 points for display

        # Current status
        self.current_metrics = {}
        self.current_prediction = {}
        self.current_anomaly = {}
        self.model_info = {}

    def fetch_health(self) -> Dict:
        """Fetch health status from API"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            print(f"Error fetching health: {e}")
            return {}

    def fetch_metrics(self) -> Dict:
        """Fetch latest telemetry metrics"""
        try:
            # Try to get recent data from local telemetry file as fallback
            telemetry_path = "/home/chandanraj-m/opsnexus-local-data/telemetry.json"
            if os.path.exists(telemetry_path):
                with open(telemetry_path, 'r') as f:
                    data = json.load(f)
                    if data:
                        latest = data[-1]  # Most recent entry
                        metrics = latest.get('metrics', {}).get('system', {})
                        # Format for dashboard
                        return {
                            'cpu_usage_percent': metrics.get('cpu', {}).get('usage_percent', 0),
                            'memory_usage_percent': metrics.get('memory', {}).get('usage_percent', 0),
                            'disk_read_mbps': metrics.get('disk', {}).get('read_mbps', 0),
                            'disk_write_mbps': metrics.get('disk', {}).get('write_mbps', 0),
                            'network_bytes_sent': metrics.get('network', {}).get('bytes_sent', 0),
                            'network_bytes_recv': metrics.get('network', {}).get('bytes_recv', 0),
                            'uptime_seconds': metrics.get('uptime', {}).get('uptime', 0),
                            'process_count': metrics.get('processes', {}).get('count', 0),
                            'timestamp': latest.get('timestamp', datetime.now().isoformat())
                        }
            # Fallback to API if local file not available
            response = requests.get(f"{self.api_base_url}/metrics/recent", timeout=5)
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            print(f"Error fetching metrics: {e}")
            return {}

    def fetch_prediction(self) -> Dict:
        """Fetch CPU usage prediction"""
        try:
            payload = {
                "agent_id": "local-agent",
                "horizon_minutes": 10,
                "lookback_points": 10
            }
            response = requests.post(
                f"{self.api_base_url}/predict/cpu",
                json=payload,
                timeout=5
            )
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            print(f"Error fetching prediction: {e}")
            return {}

    def fetch_anomaly(self) -> Dict:
        """Fetch anomaly detection status"""
        try:
            payload = {
                "agent_id": "local-agent",
                "lookback_minutes": 30,
                "sensitivity": "medium"
            }
            response = requests.post(
                f"{self.api_base_url}/detect/anomaly",
                json=payload,
                timeout=5
            )
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            print(f"Error fetching anomaly data: {e}")
            return {}

    def fetch_model_info(self) -> Dict:
        """Fetch model information"""
        try:
            response = requests.get(f"{self.api_base_url}/models/info/enhanced", timeout=5)
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            print(f"Error fetching model info: {e}")
            return {}

    def update_data(self):
        """Update all data from API sources"""
        timestamp = datetime.now()

        # Fetch all data points
        health = self.fetch_health()
        metrics = self.fetch_metrics()
        prediction = self.fetch_prediction()
        anomaly = self.fetch_anomaly()
        model_info = self.fetch_model_info()

        # Store current data
        self.current_metrics = metrics
        self.current_prediction = prediction
        self.current_anomaly = anomaly
        self.model_info = model_info

        # Update history for trending
        if metrics and 'cpu_usage_percent' in metrics:
            self.cpu_history.append(metrics['cpu_usage_percent'])
            self.timestamp_history.append(timestamp)

            # Keep only recent points
            if len(self.cpu_history) > self.max_history_points:
                self.cpu_history = self.cpu_history[-self.max_history_points:]
                self.timestamp_history = self.timestamp_history[-self.max_history_points:]

        if prediction and 'predicted_cpu_usage_percent' in prediction:
            self.prediction_history.append(prediction['predicted_cpu_usage_percent'])
            if len(self.prediction_history) > self.max_history_points:
                self.prediction_history = self.prediction_history[-self.max_history_points:]

        if anomaly and 'anomaly_score' in anomaly:
            self.anomaly_history.append(anomaly['anomaly_score'])
            if len(self.anomaly_history) > self.max_history_points:
                self.anomaly_history = self.anomaly_history[-self.max_history_points:]

    def display_text_dashboard(self):
        """Display dashboard in text format"""
        # Clear screen (works on most terminals)
        os.system('clear' if os.name == 'posix' else 'cls')

        print("=" * 80)
        print("🚀 OpsNexus-ML Real-time Intelligence Dashboard")
        print("=" * 80)
        print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Health Status
        health_status = "🟢 Healthy" if self.current_metrics.get('model_loaded', False) else "🔴 Unhealthy"
        print(f"🏥 System Health: {health_status}")
        if self.model_info:
            model_type = self.model_info.get('model_type', 'Unknown')
            features = self.model_info.get('feature_count', 0)
            print(f"🧠 Model: {model_type} ({features} features)")
        print()

        # Current Metrics
        print("📊 Current System Metrics:")
        print("-" * 40)
        if self.current_metrics:
            cpu = self.current_metrics.get('cpu_usage_percent', 0)
            memory = self.current_metrics.get('memory_usage_percent', 0)
            disk_read = self.current_metrics.get('disk_read_mbps', 0)
            disk_write = self.current_metrics.get('disk_write_mbps', 0)
            net_sent = self.current_metrics.get('network_bytes_sent', 0)
            net_recv = self.current_metrics.get('network_bytes_recv', 0)
            uptime = self.current_metrics.get('uptime_seconds', 0)
            processes = self.current_metrics.get('process_count', 0)

            # Format bytes
            def format_bytes(bytes_val):
                if bytes_val >= 1e9:
                    return f"{bytes_val/1e9:.2f} GB"
                elif bytes_val >= 1e6:
                    return f"{bytes_val/1e6:.2f} MB"
                elif bytes_val >= 1e3:
                    return f"{bytes_val/1e3:.2f} KB"
                else:
                    return f"{bytes_val:.0f} B"

            print(f"  CPU Usage:     {cpu:6.2f}%")
            print(f"  Memory Usage:  {memory:6.2f}%")
            print(f"  Disk Read:     {disk_read:8.2f} MB/s")
            print(f"  Disk Write:    {disk_write:8.2f} MB/s")
            print(f"  Network Sent:  {format_bytes(net_sent):>12}/s")
            print(f"  Network Recv:  {format_bytes(net_recv):>12}/s")
            print(f"  Uptime:        {uptime/3600:6.2f} hours")
            print(f"  Processes:     {processes:6d}")
        else:
            print("  No metrics available")
        print()

        # Prediction
        print("🔮 CPU Usage Prediction (10-min horizon):")
        print("-" * 40)
        if self.current_prediction and 'predicted_cpu_usage_percent' in self.current_prediction:
            pred = self.current_prediction['predicted_cpu_usage_percent']
            lower = self.current_prediction.get('confidence_interval', {}).get('lower', 0)
            upper = self.current_prediction.get('confidence_interval', {}).get('upper', 0)
            actual_cpu = self.current_metrics.get('cpu_usage_percent', 0) if self.current_metrics else 0

            print(f"  Predicted:     {pred:6.2f}%")
            print(f"  Actual CPU:    {actual_cpu:6.2f}%")
            print(f"  Difference:    {pred - actual_cpu:6.2f}%")
            print(f"  Confidence:    [{lower:5.2f}%, {upper:5.2f}%]")

            # Trend indicator
            if len(self.prediction_history) >= 2:
                trend = self.prediction_history[-1] - self.prediction_history[-2]
                trend_symbol = "📈" if trend > 0.1 else "📉" if trend < -0.1 else "➡️"
                print(f"  Trend:         {trend_symbol} {trend:+.2f}%")
        else:
            print("  No prediction available")
        print()

        # Anomaly Detection
        print("🚨 Anomaly Detection Status:")
        print("-" * 40)
        if self.current_anomaly and 'anomaly_score' in self.current_anomaly:
            score = self.current_anomaly['anomaly_score']
            is_anomaly = self.current_anomaly.get('is_anomaly', False)
            confidence = self.current_anomaly.get('confidence', 0)

            status = "🔴 ANOMALY DETECTED" if is_anomaly else "🟢 Normal"
            print(f"  Status:        {status}")
            print(f"  Anomaly Score: {score:6.3f}")
            print(f"  Confidence:    {confidence:6.3f}")
            print(f"  Threshold:     0.700")

            # Show contributing factors if available
            factors = self.current_anomaly.get('contributing_factors', [])
            if factors:
                print("  Top Contributing Factors:")
                for i, factor in enumerate(factors[:3]):  # Show top 3
                    feat = factor.get('feature', 'unknown')
                    dev = factor.get('deviation_score', 0)
                    val = factor.get('value', 0)
                    print(f"    {i+1}. {feat}: deviation={dev:.3f}, value={val:.2f}")
        else:
            print("  No anomaly data available")
        print()

        # Model Performance (if available)
        if self.model_info and 'training_metrics' in self.model_info:
            metrics = self.model_info['training_metrics']
            print("📈 Model Performance Metrics:")
            print("-" * 40)
            print(f"  Training MAE:  {metrics.get('train_mae', 0):.4f}")
            print(f"  Test MAE:      {metrics.get('test_mae', 0):.4f}")
            print(f"  Training R²:   {metrics.get('train_r2', 0):.4f}")
            print(f"  Test R²:       {metrics.get('test_r2', 0):.4f}")
            print()

        # History Trends (text-based sparklines)
        if len(self.cpu_history) >= 5:
            print("📉 Recent CPU Usage Trend (last 10 readings):")
            print("-" * 40)
            recent_cpu = self.cpu_history[-10:] if len(self.cpu_history) >= 10 else self.cpu_history
            if recent_cpu:
                min_val = min(recent_cpu)
                max_val = max(recent_cpu)
                range_val = max_val - min_val if max_val > min_val else 1

                sparkline = ""
                for val in recent_cpu:
                    # Normalize to 0-8 range for sparkline characters
                    normalized = int(((val - min_val) / range_val) * 8) if range_val > 0 else 4
                    sparkline += "▁▂▃▄▅▆▇█"[normalized]

                print(f"  {sparkline}")
                print(f"  Range: {min_val:.1f}% - {max_val:.1f}%")
        print()

        # Instructions
        print("💡 Controls:")
        print("  Press Ctrl+C to exit the dashboard")
        print("  Data refreshes every {} seconds".format(self.refresh_interval))
        print("=" * 80)

    def run(self):
        """Run the dashboard"""
        print("Starting OpsNexus-ML Real-time Dashboard...")
        print(f"Connecting to API at {self.api_base_url}")
        print("Press Ctrl+C to exit")
        time.sleep(2)

        self.running = True

        try:
            while self.running:
                self.update_data()
                self.display_text_dashboard()
                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            print("\n\n👋 Dashboard stopped by user")
            self.running = False
        except Exception as e:
            print(f"\n\n❌ Dashboard error: {e}")
            self.running = False

def main():
    """Main function to run the dashboard"""
    import argparse

    parser = argparse.ArgumentParser(description='OpsNexus-ML Real-time Dashboard')
    parser.add_argument('--api-url', default='http://localhost:5000',
                       help='Base URL for the OpsNexus-ML API (default: http://localhost:5000)')
    parser.add_argument('--interval', type=int, default=5,
                       help='Refresh interval in seconds (default: 5)')

    args = parser.parse_args()

    dashboard = OpsNexusMLDashboard(
        api_base_url=args.api_url,
        refresh_interval=args.interval
    )

    dashboard.run()

if __name__ == "__main__":
    main()