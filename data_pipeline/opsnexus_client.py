"""
OpsNexus API Integration Client
Handles communication with OpsNexus platform to fetch telemetry data
"""
import json
import time
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging
from urllib import request, error
import ssl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpsNexusClient:
    """
    Client for fetching telemetry data from OpsNexus platform
    """

    def __init__(self,
                 base_url: str,
                 api_key: Optional[str] = None,
                 timeout: int = 30,
                 max_retries: int = 3,
                 backoff_factor: float = 0.3):
        """
        Initialize the OpsNexus API client

        Args:
            base_url: Base URL of OpsNexus platform (e.g., https://opsnexus.example.com)
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            backoff_factor: Backoff factor for exponential backoff
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        # Create SSL context that doesn't verify certificates (for dev environments)
        # In production, you should use proper certificate validation
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

    def _make_request(self, url: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make HTTP GET request with retry logic

        Args:
            url: URL to request
            params: Query parameters

        Returns:
            Parsed JSON response
        """
        # Build full URL with parameters
        if params:
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            url = f"{url}?{query_string}"

        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Making request to {url} (attempt {attempt + 1})")

                req = request.Request(url)
                if self.api_key:
                    req.add_header('Authorization', f'Bearer {self.api_key}')
                req.add_header('Content-Type', 'application/json')
                req.add_header('User-Agent', 'OpsNexus-ML-Client/1.0')

                # Make request
                response = request.urlopen(req, timeout=self.timeout, context=self.ssl_context)

                # Read and decode response
                data = response.read().decode('utf-8')
                json_data = json.loads(data)

                logger.info(f"Request successful: {len(json_data) if isinstance(json_data, list) else 'object'} items")
                return json_data

            except error.HTTPError as e:
                logger.warning(f"HTTP error {e.code}: {e.reason}")
                if e.code in [429, 500, 502, 503, 504] and attempt < self.max_retries:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise
            except error.URLError as e:
                logger.warning(f"URL error: {e.reason}")
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON: {e}")
                raise
            except Exception as e:
                logger.warning(f"Unexpected error: {e}")
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise

        # This should never be reached due to the raise in the loop
        raise Exception("Max retries exceeded")

    def fetch_agent_analytics(self,
                             agent_id: str,
                             lookback_minutes: int = 60,
                             metrics: Optional[List[str]] = None,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Fetch telemetry data for a specific agent from OpsNexus analytics endpoint

        Args:
            agent_id: ID of the agent to fetch data for
            lookback_minutes: How many minutes of data to look back (if start/end not provided)
            metrics: Specific metrics to fetch (if None, fetches all available)
            start_time: Start time for data range (overrides lookback_minutes if provided)
            end_time: End time for data range (defaults to now if not provided)

        Returns:
            List of telemetry records in OpsNexus MetricPayload format
        """
        # Calculate time range
        if end_time is None:
            end_time = datetime.utcnow()

        if start_time is None:
            start_time = end_time - timedelta(minutes=lookback_minutes)

        # Build request parameters
        params = {
            'agent_id': agent_id,
            'start_time': start_time.isoformat() + 'Z',
            'end_time': end_time.isoformat() + 'Z'
        }

        if metrics:
            params['metrics'] = ','.join(metrics)

        # Construct endpoint URL
        endpoint = f"{self.base_url}/api/v1/agents/{agent_id}/analytics"

        try:
            data = self._make_request(endpoint, params)

            # Handle different possible response formats
            if isinstance(data, dict):
                # If response is wrapped in a standard API response
                if 'data' in data:
                    telemetry_data = data['data']
                elif 'telemetry' in data:
                    telemetry_data = data['telemetry']
                else:
                    telemetry_data = data
            elif isinstance(data, list):
                telemetry_data = data
            else:
                raise ValueError(f"Unexpected response format: {type(data)}")

            logger.info(f"Successfully fetched {len(telemetry_data)} telemetry records for agent {agent_id}")
            return telemetry_data

        except Exception as e:
            logger.error(f"Failed to fetch analytics for agent {agent_id}: {str(e)}")
            raise

    def fetch_multiple_agents(self,
                             agent_ids: List[str],
                             lookback_minutes: int = 60,
                             metrics: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch telemetry data for multiple agents

        Args:
            agent_ids: List of agent IDs to fetch data for
            lookback_minutes: How many minutes of data to look back
            metrics: Specific metrics to fetch

        Returns:
            Dictionary mapping agent_id to list of telemetry records
        """
        results = {}

        for agent_id in agent_ids:
            try:
                data = self.fetch_agent_analytics(
                    agent_id=agent_id,
                    lookback_minutes=lookback_minutes,
                    metrics=metrics
                )
                results[agent_id] = data
            except Exception as e:
                logger.error(f"Failed to fetch data for agent {agent_id}: {str(e)}")
                results[agent_id] = []  # Empty list for failed agents

        return results

    def validate_metric_payload(self, payload: Dict[str, Any]) -> bool:
        """
        Validate that a payload matches the expected OpsNexus MetricPayload structure

        Args:
            payload: Payload to validate

        Returns:
            True if payload is valid, False otherwise
        """
        try:
            # Check required top-level fields
            required_fields = ['agent_id', 'timestamp', 'metrics']
            for field in required_fields:
                if field not in payload:
                    logger.warning(f"Missing required field '{field}' in payload")
                    return False

            # Check metrics structure
            if not isinstance(payload['metrics'], dict):
                logger.warning("Metrics field must be a dictionary")
                return False

            if 'system' not in payload['metrics']:
                logger.warning("Missing 'system' in metrics")
                return False

            system = payload['metrics']['system']
            required_system_fields = ['cpu', 'memory', 'disk', 'network', 'uptime', 'processes']
            for field in required_system_fields:
                if field not in system:
                    logger.warning(f"Missing '{field}' in system metrics")
                    return False

            # Check CPU structure
            cpu = system['cpu']
            if not isinstance(cpu, dict):
                logger.warning("CPU metrics must be a dictionary")
                return False

            required_cpu_fields = ['usage_percent', 'per_cpu', 'count']
            for field in required_cpu_fields:
                if field not in cpu:
                    logger.warning(f"Missing '{field}' in CPU metrics")
                    return False

            # Check memory structure
            memory = system['memory']
            if not isinstance(memory, dict):
                logger.warning("Memory metrics must be a dictionary")
                return False

            required_memory_fields = ['usage_percent', 'available_mb', 'used_mb', 'free_mb']
            for field in required_memory_fields:
                if field not in memory:
                    logger.warning(f"Missing '{field}' in memory metrics")
                    return False

            # Basic type checks
            if not isinstance(payload['agent_id'], str):
                logger.warning("agent_id must be a string")
                return False

            # Try to parse timestamp
            try:
                datetime.fromisoformat(payload['timestamp'].replace('Z', '+00:00'))
            except ValueError:
                logger.warning("Invalid timestamp format")
                return False

            return True

        except Exception as e:
            logger.warning(f"Error validating metric payload: {str(e)}")
            return False

    def close(self):
        """Close the session (nothing to close for urllib)"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience function for quick usage
def create_opsnexus_client(base_url: str,
                          api_key: Optional[str] = None,
                          **kwargs) -> OpsNexusClient:
    """
    Convenience function to create an OpsNexus client

    Args:
        base_url: Base URL of OpsNexus platform
        api_key: Optional API key for authentication
        **kwargs: Additional arguments passed to OpsNexusClient constructor

    Returns:
        Configured OpsNexusClient instance
    """
    return OpsNexusClient(base_url=base_url, api_key=api_key, **kwargs)


if __name__ == "__main__":
    # Example usage and basic test
    print("OpsNexus Client Module - Testing Import")
    print("=" * 40)

    # Test that the class can be imported and instantiated
    try:
        client = OpsNexusClient(
            base_url="https://example.com",
            api_key="test-key"
        )
        print("✓ Client creation successful")
        print(f"✓ Base URL: {client.base_url}")
        print(f"✓ Has API key: {bool(client.api_key)}")
        client.close()
        print("✓ Client cleanup successful")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("\nModule loaded successfully!")