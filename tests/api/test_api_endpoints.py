"""
API endpoint tests for OpsNexus-ML
"""
import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import the Flask app
from api.app import app


@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestHealthEndpoint:
    """Test /health endpoint"""

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert data['service'] == 'OpsNexus-ML API'
        assert 'model_loaded' in data
        assert 'timestamp' in data


class TestPredictCpuEndpoint:
    """Test /predict/cpu endpoint"""

    @patch('api.app.cpu_predictor')
    @patch('api.app.model_loaded', True)
    @patch('api.app.data_pipeline')
    def test_predict_cpu_success(self, mock_pipeline, mock_predictor, client):
        """Test successful CPU prediction"""
        # Mock predictor
        mock_predictor.predict_next_cpu_usage.return_value = 65.5
        mock_predictor.feature_names = [f'f_{i}' for i in range(20)]

        # Mock pipeline
        mock_pipeline.load_data.return_value = MagicMock()
        mock_pipeline.clean_data.return_value = MagicMock()
        mock_pipeline.engineer_features.return_value = MagicMock()
        mock_pipeline.engineer_features.return_value.columns = [f'f_{i}' for i in range(20)] + ['agent_id', 'target']
        mock_pipeline.engineer_features.return_value.__getitem__ = lambda self, key: MagicMock()
        mock_pipeline.engineer_features.return_value.iloc = MagicMock()
        mock_pipeline.engineer_features.return_value.iloc.__getitem__ = lambda self, key: MagicMock(values=[[0]*20])

        response = client.post('/predict/cpu',
                               data=json.dumps({
                                   'agent_id': 'agent-web-01',
                                   'horizon_minutes': 15,
                                   'lookback_points': 100
                               }),
                               content_type='application/json')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['agent_id'] == 'agent-web-01'
        assert data['predicted_cpu_usage_percent'] == 65.5
        assert 'confidence_interval' in data
        assert 'model_info' in data

    def test_predict_cpu_model_not_loaded(self, client):
        """Test CPU prediction when model not loaded"""
        with patch('api.app.model_loaded', False):
            response = client.post('/predict/cpu',
                                   data=json.dumps({'agent_id': 'agent-web-01'}),
                                   content_type='application/json')

            assert response.status_code == 503
            data = json.loads(response.data)
            assert data['error'] == 'Model not loaded'

    def test_predict_cpu_invalid_json(self, client):
        """Test CPU prediction with invalid JSON"""
        with patch('api.app.model_loaded', True):
            response = client.post('/predict/cpu',
                                   data='invalid json',
                                   content_type='application/json')

            assert response.status_code == 500


class TestDetectAnomalyEndpoint:
    """Test /detect/anomaly endpoint"""

    @patch('api.app.isolation_forest_detector')
    @patch('api.app.model_loaded', True)
    @patch('api.app.data_pipeline')
    def test_detect_anomaly_success(self, mock_pipeline, mock_detector, client):
        """Test successful anomaly detection"""
        # Mock detector
        mock_detector.predict_anomaly_score.return_value = [0.75]
        mock_detector.detect_anomaly.return_value = [True]
        mock_detector.explain_anomaly.return_value = {
            'anomaly_score': 0.75,
            'top_contributing_factors': [
                {'feature': 'cpu_usage_percent', 'deviation_score': 2.1, 'value': 95.0, 'importance': 0.3},
                {'feature': 'memory_usage_percent', 'deviation_score': 1.8, 'value': 88.0, 'importance': 0.25}
            ]
        }
        mock_detector.estimator.n_estimators = 100
        mock_detector.get_feature_importance.return_value = {'cpu_usage_percent': 0.3, 'memory_usage_percent': 0.25}

        # Mock pipeline
        mock_pipeline.load_data.return_value = MagicMock()
        mock_pipeline.clean_data.return_value = MagicMock()
        mock_pipeline.engineer_features.return_value = MagicMock()
        mock_pipeline.engineer_features.return_value.columns = ['cpu_usage_percent', 'memory_usage_percent', 'agent_id']
        mock_pipeline.engineer_features.return_value.__getitem__ = lambda self, key: MagicMock()
        mock_pipeline.engineer_features.return_value.iloc = MagicMock()
        mock_pipeline.engineer_features.return_value.iloc.__getitem__ = lambda self, key: MagicMock(values=[[95, 88]])
        mock_pipeline.engineer_features.return_value.mean.return_value = 50
        mock_pipeline.engineer_features.return_value.std.return_value = 20

        response = client.post('/detect/anomaly',
                               data=json.dumps({
                                   'agent_id': 'agent-db-02',
                                   'lookback_minutes': 30,
                                   'sensitivity': 'high'
                               }),
                               content_type='application/json')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['agent_id'] == 'agent-db-02'
        assert data['anomaly_score'] == 0.75
        assert data['is_anomaly'] is True
        assert 'contributing_factors' in data
        assert 'model_info' in data

    def test_detect_anomaly_models_not_loaded(self, client):
        """Test anomaly detection when models not loaded"""
        with patch('api.app.model_loaded', False), \
             patch('api.app.isolation_forest_detector', None):
            response = client.post('/detect/anomaly',
                                   data=json.dumps({'agent_id': 'agent-db-02'}),
                                   content_type='application/json')

            assert response.status_code == 503
            data = json.loads(response.data)
            assert data['error'] == 'Models not loaded'


class TestModelInfoEndpoint:
    """Test /models/info endpoint"""

    @patch('api.app.cpu_predictor')
    @patch('api.app.model_loaded', True)
    @patch('api.app.feature_names', ['f1', 'f2', 'f3'])
    def test_model_info_success(self, mock_predictor, client):
        """Test successful model info retrieval"""
        mock_predictor.get_feature_importance.return_value = {
            'f1': 0.5,
            'f2': 0.3,
            'f3': 0.2,
            'intercept': 10.0
        }

        response = client.get('/models/info')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['model_loaded'] is True
        assert data['model_type'] == 'linear_regression'
        assert data['feature_count'] == 3
        assert 'top_features' in data
        assert 'intercept' in data

    def test_model_info_not_loaded(self, client):
        """Test model info when no model loaded"""
        with patch('api.app.model_loaded', False), \
             patch('api.app.cpu_predictor', None):
            response = client.get('/models/info')

            assert response.status_code == 404
            data = json.loads(response.data)
            assert data['error'] == 'No model loaded'


class TestErrorHandling:
    """Test error handling in API endpoints"""

    def test_404_endpoint(self, client):
        """Test 404 for unknown endpoint"""
        response = client.get('/unknown/endpoint')
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test 405 for wrong HTTP method"""
        response = client.put('/health')
        assert response.status_code == 405


if __name__ == '__main__':
    pytest.main([__file__, '-v'])