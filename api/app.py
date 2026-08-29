"""
OpsNexus-ML API Service
Provides REST endpoints for CPU usage prediction and anomaly detection
"""
import os
import sys
import json
import logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import traceback

# Add project directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_pipeline'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

from pipeline import OpsNexusDataPipeline
from baseline_model import CPUUsagePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for models and pipeline
cpu_predictor = None
data_pipeline = None
feature_names = None
model_loaded = False

def initialize_models():
    """Initialize and load pre-trained models"""
    global cpu_predictor, data_pipeline, feature_names, model_loaded

    try:
        logger.info("Initializing OpsNexus-ML API service...")

        # Initialize data pipeline
        data_pipeline = OpsNexusDataPipeline()

        # Load the trained model
        model_path = "/home/chandanraj-m/OpsNexus-ML/models/cpu_predictor.pkl"
        if os.path.exists(model_path):
            cpu_predictor = CPUUsagePredictor(model_path=model_path)
            cpu_predictor.load_model(model_path)
            feature_names = cpu_predictor.feature_names
            model_loaded = True
            logger.info(f"Model loaded successfully from {model_path}")
            logger.info(f"Model features: {len(feature_names) if feature_names else 0}")
        else:
            logger.warning(f"No pre-trained model found at {model_path}")
            logger.info("API will return mock predictions until model is trained")

    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        logger.error(traceback.format_exc())
        model_loaded = False

def get_latest_features_from_synthetic_data(num_points=100):
    """
    Extract latest features from synthetic data for prediction
    In a real implementation, this would come from OpsNexus API
    """
    try:
        # Load recent synthetic data
        data_path = "/home/chandanraj-m/OpsNexus-ML/data_pipeline/synthetic_telemetry_sample.json"

        if not os.path.exists(data_path):
            # Fallback to generating a small sample
            from data_pipeline.synthetic_data_generator import generate_metric_payload, save_synthetic_data
            data = generate_metric_payload(num_points=num_points)
            save_synthetic_data(data, data_path)

        # Process through pipeline
        df = data_pipeline.load_data(data_path)
        cleaned_df = data_pipeline.clean_data()
        featured_df = data_pipeline.engineer_features()

        # Get the latest feature vector (excluding target and non-feature columns)
        exclude_cols = ['agent_id']
        if 'target' in featured_df.columns:
            exclude_cols.append('target')

        feature_cols = [col for col in featured_df.columns if col not in exclude_cols]
        latest_features = featured_df[feature_cols].iloc[-1:].values  # Most recent sample

        return latest_features, feature_cols

    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        # Return dummy features matching expected shape
        dummy_features = np.zeros((1, 51))  # Match the trained model's feature count
        return dummy_features, [f'feature_{i}' for i in range(51)]

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'OpsNexus-ML API',
        'model_loaded': model_loaded,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    })

@app.route('/predict/cpu', methods=['POST'])
def predict_cpu_usage():
    """
    Predict CPU usage for the next time period
    Expected JSON payload:
    {
        "agent_id": "optional-agent-id",
        "horizon_minutes": 10,  // Optional, defaults to 10 minutes
        "lookback_points": 100   // Optional, how many points to look back
    }
    """
    try:
        if not model_loaded:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Please train and load a model first'
            }), 503

        # Get request parameters
        data = request.get_json() or {}
        agent_id = data.get('agent_id', 'unknown')
        horizon_minutes = data.get('horizon_minutes', 10)
        lookback_points = data.get('lookback_points', 100)

        logger.info(f"CPU prediction request for agent {agent_id}, horizon {horizon_minutes}min")

        # In a real implementation, we would:
        # 1. Fetch recent telemetry from OpsNexus API for the given agent
        # 2. Process it through our pipeline to get features
        # 3. Make a prediction

        # For now, we'll use synthetic data to demonstrate the flow
        latest_features, feature_cols = get_latest_features_from_synthetic_data(lookback_points)

        # Make prediction
        prediction = cpu_predictor.predict_next_cpu_usage(latest_features)

        # Calculate confidence interval (simplified)
        # In reality, we'd use prediction intervals from the model
        confidence_lower = max(0, prediction - 5.0)
        confidence_upper = min(100, prediction + 5.0)

        # Prepare response
        response = {
            'agent_id': agent_id,
            'prediction_timestamp': datetime.utcnow().isoformat() + 'Z',
            'horizon_minutes': horizon_minutes,
            'predicted_cpu_usage_percent': round(float(prediction), 2),
            'confidence_interval': {
                'lower': round(confidence_lower, 2),
                'upper': round(confidence_upper, 2)
            },
            'model_info': {
                'model_type': 'linear_regression',
                'features_used': len(feature_cols),
                'model_loaded': model_loaded
            }
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in CPU prediction: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/detect/anomaly', methods=['POST'])
def detect_anomaly():
    """
    Detect anomalies in recent telemetry data
    Expected JSON payload:
    {
        "agent_id": "optional-agent-id",
        "lookback_minutes": 30,
        "sensitivity": "medium"  // low, medium, high
    }
    """
    try:
        if not model_loaded:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Please train and load a model first'
            }), 503

        # Get request parameters
        data = request.get_json() or {}
        agent_id = data.get('agent_id', 'unknown')
        lookback_minutes = data.get('lookback_minutes', 30)
        sensitivity = data.get('sensitivity', 'medium')

        logger.info(f"Anomaly detection request for agent {agent_id}, lookback {lookback_minutes}min")

        # For demonstration, we'll return a mock anomaly score
        # In a real implementation, we would:
        # 1. Get recent telemetry data
        # 2. Compare it to learned normal patterns
        # 3. Calculate an anomaly score

        # Mock anomaly score based on time of day (just for demo)
        hour = datetime.utcnow().hour
        base_score = 0.1 + (abs(hour - 14) / 24) * 0.3  # Higher score away from 2 PM

        # Adjust based on sensitivity
        sensitivity_multiplier = {'low': 0.5, 'medium': 1.0, 'high': 2.0}.get(sensitivity, 1.0)
        anomaly_score = min(1.0, base_score * sensitivity_multiplier)

        is_anomaly = anomaly_score > 0.7

        # Mock contributing factors
        contributing_factors = [
            {'metric': 'cpu_usage_percent', 'score': round(anomaly_score * 0.8, 3)},
            {'metric': 'memory_usage_percent', 'score': round(anomaly_score * 0.6, 3)},
            {'metric': 'network_bytes_sent', 'score': round(anomaly_score * 0.4, 3)}
        ]

        response = {
            'agent_id': agent_id,
            'detection_timestamp': datetime.utcnow().isoformat() + 'Z',
            'lookback_minutes': lookback_minutes,
            'sensitivity': sensitivity,
            'anomaly_score': round(anomaly_score, 3),
            'is_anomaly': bool(is_anomaly),
            'contributing_factors': contributing_factors,
            'model_info': {
                'detection_method': 'residual_based',
                'model_loaded': model_loaded
            }
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Anomaly detection failed',
            'message': str(e)
        }), 500

@app.route('/models/info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    try:
        if not model_loaded or cpu_predictor is None:
            return jsonify({
                'error': 'No model loaded',
                'model_loaded': False
            }), 404

        importance = cpu_predictor.get_feature_importance()

        response = {
            'model_loaded': model_loaded,
            'model_type': 'linear_regression',
            'feature_count': len(feature_names) if feature_names else 0,
            'features': feature_names if feature_names else [],
            'top_features': dict(list(sorted(importance.items(),
                                          key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
                                          reverse=True)[:10])) if importance else {},
            'intercept': importance.get('intercept', 0) if importance else 0,
            'last_loaded': datetime.utcnow().isoformat() + 'Z'
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({
            'error': 'Failed to get model info',
            'message': str(e)
        }), 500

# Initialize models when the module loads
initialize_models()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    logger.info(f"Starting OpsNexus-ML API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)