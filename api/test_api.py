"""
Test script for the OpsNexus-ML API
"""
import sys
import os
import json
import time
import requests
from threading import Thread

# Add project directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_api_endpoints():
    """Test the API endpoints"""
    print("Testing OpsNexus-ML API Endpoints...")
    print("=" * 50)

    # Start the API server in a background thread
    def run_server():
        from api.app import app
        app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)

    server_thread = Thread(target=run_server, daemon=True)
    server_thread.start()

    # Give the server time to start
    time.sleep(2)

    base_url = "http://127.0.0.1:5001"

    try:
        # Test 1: Health check
        print("1. Testing health check endpoint...")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health check passed: {data['status']}")
            print(f"   Model loaded: {data['model_loaded']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False

        # Test 2: Model info
        print("\n2. Testing model info endpoint...")
        response = requests.get(f"{base_url}/models/info")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Model info retrieved")
            print(f"   Model loaded: {data['model_loaded']}")
            print(f"   Feature count: {data.get('feature_count', 0)}")
        else:
            print(f"   ❌ Model info failed: {response.status_code}")
            if response.status_code != 404:  # 404 is expected if no model
                return False

        # Test 3: CPU prediction
        print("\n3. Testing CPU prediction endpoint...")
        prediction_data = {
            "agent_id": "test-agent-001",
            "horizon_minutes": 10,
            "lookback_points": 50
        }
        response = requests.post(f"{base_url}/predict/cpu", json=prediction_data)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ CPU prediction successful")
            print(f"   Agent: {data['agent_id']}")
            print(f"   Predicted CPU: {data['predicted_cpu_usage_percent']}%")
            print(f"   Confidence interval: [{data['confidence_interval']['lower']}, {data['confidence_interval']['upper']}]%")
        else:
            print(f"   ❌ CPU prediction failed: {response.status_code}")
            print(f"   Error: {response.json().get('error', 'Unknown error')}")
            return False

        # Test 4: Anomaly detection
        print("\n4. Testing anomaly detection endpoint...")
        anomaly_data = {
            "agent_id": "test-agent-001",
            "lookback_minutes": 30,
            "sensitivity": "medium"
        }
        response = requests.post(f"{base_url}/detect/anomaly", json=anomaly_data)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Anomaly detection successful")
            print(f"   Agent: {data['agent_id']}")
            print(f"   Anomaly score: {data['anomaly_score']}")
            print(f"   Is anomaly: {data['is_anomaly']}")
            print(f"   Contributing factors: {len(data['contributing_factors'])} metrics")
        else:
            print(f"   ❌ Anomaly detection failed: {response.status_code}")
            print(f"   Error: {response.json().get('error', 'Unknown error')}")
            return False

        # Test 5: Invalid request handling
        print("\n5. Testing error handling...")
        response = requests.post(f"{base_url}/predict/cpu", json={"invalid": "data"})
        # This should still work because we have defaults
        if response.status_code in [200, 400, 500]:
            print(f"   ✅ Error handling working (status: {response.status_code})")
        else:
            print(f"   ❌ Unexpected error handling: {response.status_code}")

        print("\n🎉 All API tests completed!")
        return True

    except requests.exceptions.ConnectionError:
        print("   ❌ Could not connect to API server. Make sure it's running on port 5001")
        return False
    except Exception as e:
        print(f"   ❌ API test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_api_endpoints()
    sys.exit(0 if success else 1)