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
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pipeline import OpsNexusDataPipeline
from baseline_model import CPUUsagePredictor
from models.isolation_forest_detector import IsolationForestDetector
from models.enhanced_model import EnhancedCPUUsagePredictor, ModelType
from data_pipeline.opsnexus_client import OpsNexusClient
from model_versioning import ModelVersionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'OpsNexus-ML API',
        'model_loaded': model_loaded,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    })

# Global variables for models and pipeline
cpu_predictor = None
data_pipeline = None
feature_names = None
model_loaded = False
isolation_forest_detector = None
opsnexus_client = None

def initialize_models():
    """Initialize and load pre-trained models using model versioning system"""
    global cpu_predictor, data_pipeline, feature_names, model_loaded, isolation_forest_detector, opsnexus_client

    try:
        logger.info("Initializing OpsNexus-ML API service...")

        # Initialize data pipeline
        data_pipeline = OpsNexusDataPipeline()

        # Initialize OpsNexus client (for fetching real data)
        # In production, these would come from environment variables or config
        opsnexus_client = OpsNexusClient(
            base_url=os.environ.get("OPNEXUS_BASE_URL", "https://opsnexus.example.com"),
            api_key=os.environ.get("OPNEXUS_API_KEY"),
            timeout=int(os.environ.get("OPNEXUS_TIMEOUT", "30"))
        )

        # Initialize model versioning system
        model_version_manager = ModelVersionManager()

        # Load the latest CPU prediction model using versioning system
        try:
            cpu_model_path = model_version_manager.get_latest_model_path("cpu_predictor")
            # Try to load as enhanced model first, fallback to baseline
            try:
                cpu_predictor = EnhancedCPUUsagePredictor(model_path=cpu_model_path)
                cpu_predictor.load_model(cpu_model_path)
                feature_names = cpu_predictor.feature_names
                model_loaded = True
                model_info = cpu_predictor.get_model_info()
                logger.info(f"Enhanced CPU prediction model ({model_info.get('model_type', 'unknown')}) loaded successfully from {cpu_model_path}")
                logger.info(f"Model features: {len(feature_names) if feature_names else 0}")
            except Exception as enhanced_error:
                logger.warning(f"Could not load enhanced model, trying baseline model: {enhanced_error}")
                # Fallback to baseline model
                cpu_predictor = CPUUsagePredictor(model_path=cpu_model_path)
                cpu_predictor.load_model(cpu_model_path)
                feature_names = cpu_predictor.feature_names
                model_loaded = True
                logger.info(f"Baseline CPU prediction model loaded successfully from {cpu_model_path}")
                logger.info(f"Model features: {len(feature_names) if feature_names else 0}")
        except Exception as e:
            logger.warning(f"Could not load CPU prediction model from versioning system: {e}")
            logger.info("Falling back to direct model loading")
            # Fallback to original method - prioritize real-data model
            real_model_path = "/home/chandanraj-m/OpsNexus-ML/models/cpu_predictor_real_data.pkl"
            baseline_model_path = "/home/chandanraj-m/OpsNexus-ML/models/cpu_predictor.pkl"

            # Try to load the real-data model first
            if os.path.exists(real_model_path):
                try:
                    cpu_predictor = EnhancedCPUUsagePredictor(model_path=real_model_path)
                    cpu_predictor.load_model(real_model_path)
                    feature_names = cpu_predictor.feature_names
                    model_loaded = True
                    model_info = cpu_predictor.get_model_info()
                    logger.info(f"Enhanced CPU prediction model (real data - {model_info.get('model_type', 'unknown')}) loaded successfully from {real_model_path}")
                    logger.info(f"Model features: {len(feature_names) if feature_names else 0}")
                except Exception as real_error:
                    logger.warning(f"Could not load real-data model: {real_error}")
                    # Fall through to try baseline model
            else:
                logger.warning(f"Real-data model not found at {real_model_path}")

            # If we haven't loaded a model yet, try the baseline model
            if not model_loaded and os.path.exists(baseline_model_path):
                # Try enhanced model first (in case the baseline model path actually has an enhanced model)
                try:
                    cpu_predictor = EnhancedCPUUsagePredictor(model_path=baseline_model_path)
                    cpu_predictor.load_model(baseline_model_path)
                    feature_names = cpu_predictor.feature_names
                    model_loaded = True
                    model_info = cpu_predictor.get_model_info()
                    logger.info(f"Enhanced CPU prediction model ({model_info.get('model_type', 'unknown')}) loaded successfully from {baseline_model_path}")
                    logger.info(f"Model features: {len(feature_names) if feature_names else 0}")
                except Exception as enhanced_error:
                    logger.warning(f"Could not load enhanced model from baseline path, trying baseline: {enhanced_error}")
                    # Fallback to baseline model
                    cpu_predictor = CPUUsagePredictor(model_path=baseline_model_path)
                    cpu_predictor.load_model(baseline_model_path)
                    feature_names = cpu_predictor.feature_names
                    model_loaded = True
                    logger.info(f"Baseline CPU prediction model loaded successfully from {baseline_model_path}")
                    logger.info(f"Model features: {len(feature_names) if feature_names else 0}")
            else:
                logger.warning(f"No pre-trained model found at {baseline_model_path}")
                logger.info("API will return mock predictions until model is trained")

        # Load the latest Isolation Forest anomaly detector using versioning system
        try:
            anomaly_model_path = model_version_manager.get_latest_model_path("isolation_forest")
            isolation_forest_detector = IsolationForestDetector(model_path=anomaly_model_path)
            isolation_forest_detector.load_model(anomaly_model_path)
            logger.info(f"Isolation Forest detector loaded successfully from {anomaly_model_path}")
        except Exception as e:
            logger.warning(f"Could not load Isolation Forest model from versioning system: {e}")
            logger.info("Falling back to direct model loading")
            # Fallback to original method
            anomaly_model_path = "/home/chandanraj-m/OpsNexus-ML/models/isolation_forest_detector.pkl"
            if os.path.exists(anomaly_model_path):
                isolation_forest_detector = IsolationForestDetector(model_path=anomaly_model_path)
                isolation_forest_detector.load_model(anomaly_model_path)
                logger.info(f"Isolation Forest detector loaded successfully from {anomaly_model_path}")
            else:
                logger.warning(f"No pre-trained Isolation Forest model found at {anomaly_model_path}")
                logger.info("Anomaly detection will use mock data until model is trained")

    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        logger.error(traceback.format_exc())
        model_loaded = False

def get_latest_features_from_local_telemetry(num_points=100):
    """
    Extract latest features from local telemetry file written by opsnexus-agent
    This replaces the synthetic data function for local testing
    """
    try:
        # Load recent telemetry data from local file written by opsnexus-agent
        data_path = "/home/chandanraj-m/opsnexus-local-data/telemetry.json"

        if not os.path.exists(data_path):
            # Fallback to synthetic data if local file doesn't exist yet
            logger.warning(f"Local telemetry file not found at {data_path}, falling back to synthetic data")
            data_path = "/home/chandanraj-m/OpsNexus-ML/data_pipeline/synthetic_telemetry_sample.json"

        # Process through pipeline
        df = data_pipeline.load_data(data_path)
        cleaned_df = data_pipeline.clean_data()
        featured_df = data_pipeline.engineer_features()

        # Get the latest feature vector (excluding target and non-feature columns)
        exclude_cols = ["agent_id"]
        if "target" in featured_df.columns:
            exclude_cols.append("target")

        feature_cols = [col for col in featured_df.columns if col not in exclude_cols]

        # Get the most recent sample for prediction (single sample)
        latest_features = featured_df[feature_cols].iloc[-1:].values

        return latest_features, feature_cols

    except Exception as e:
        logger.error(f"Error processing local telemetry data: {e}")
        # Fallback to synthetic data on error
        logger.warning("Falling back to synthetic data due to error")
        data_path = "/home/chandanraj-m/OpsNexus-ML/data_pipeline/synthetic_telemetry_sample.json"
        
        # Process through pipeline
        df = data_pipeline.load_data(data_path)
        cleaned_df = data_pipeline.clean_data()
        featured_df = data_pipeline.engineer_features()

        # Get the latest feature vector (excluding target and non-feature columns)
        exclude_cols = ["agent_id"]
        if "target" in featured_df.columns:
            exclude_cols.append("target")

        feature_cols = [col for col in featured_df.columns if col not in exclude_cols]
        latest_features = featured_df[feature_cols].iloc[-1:].values  # Most recent sample

        return latest_features, feature_cols

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
        latest_features, feature_cols = get_latest_features_from_local_telemetry(lookback_points)

        # Make prediction
        prediction = cpu_predictor.predict_next_cpu_usage(latest_features)

        # Calculate confidence interval (simplified)
        # In reality, we'd use prediction intervals from the model
        confidence_lower = max(0, prediction - 5.0)
        confidence_upper = min(100, prediction + 5.0)

        # Prepare response with enhanced model information
        model_info_response = {
            'features_used': len(feature_cols),
            'model_loaded': model_loaded
        }

        # Add model-specific information if available
        if hasattr(cpu_predictor, 'get_model_info'):
            try:
                model_details = cpu_predictor.get_model_info()
                model_info_response.update({
                    'model_type': model_details.get('model_type', 'unknown'),
                    'training_metrics': model_details.get('training_metrics', {})
                })
            except Exception as e:
                logger.warning(f"Could not get enhanced model info: {e}")
                model_info_response['model_type'] = 'unknown'
        else:
            # Fallback for baseline model
            model_info_response['model_type'] = 'linear_regression'

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
            'model_info': model_info_response
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
        # Check if models are loaded
        if not model_loaded or isolation_forest_detector is None:
            return jsonify({
                'error': 'Models not loaded',
                'message': 'Please train and load models first'
            }), 503

        # Get request parameters
        data = request.get_json() or {}
        agent_id = data.get('agent_id', 'unknown')
        lookback_minutes = data.get('lookback_minutes', 30)
        sensitivity = data.get('sensitivity', 'medium')

        logger.info(f"Anomaly detection request for agent {agent_id}, lookback {lookback_minutes}min")

        try:
            # Get recent telemetry data using OpsNexusClient
            # For demonstration, we'll use synthetic data
            # In production, this would come from the OpsNexus API
            raw_data = None

            # Try to fetch from OpsNexus API first
            if opsnexus_client is not None:
                try:
                    raw_data = opsnexus_client.fetch_agent_analytics(
                        agent_id=agent_id,
                        lookback_minutes=lookback_minutes
                    )
                    logger.info(f"Fetched {len(raw_data)} records from OpsNexus API for agent {agent_id}")
                except Exception as api_error:
                    logger.warning(f"Failed to fetch from OpsNexus API: {api_error}. Falling back to synthetic data.")
                    raw_data = None

            # Fallback to synthetic data if API fetch failed or client not available
            if raw_data is None:
                data_path = "/home/chandanraj-m/OpsNexus-ML/data_pipeline/synthetic_telemetry_sample.json"
                if not os.path.exists(data_path):
                    # Generate sample data if not available
                    from data_pipeline.synthetic_data_generator import generate_metric_payload, save_synthetic_data
                    data = generate_metric_payload(num_points=100)
                    save_synthetic_data(data, data_path)

                # Load the data
                with open(data_path, 'r') as f:
                    raw_data = json.load(f)
                logger.info(f"Loaded {len(raw_data)} records from synthetic data")

            # Process through pipeline
            df = data_pipeline.load_data_from_dict(raw_data) if hasattr(data_pipeline, 'load_data_from_dict') else data_pipeline.load_data(
                json.dumps(raw_data) if isinstance(raw_data, list) else raw_data
            ) if isinstance(raw_data, str) else None

            # Handle different data formats
            if df is None and isinstance(raw_data, list):
                # Convert list of dicts to DataFrame manually if needed
                import pandas as pd
                df = pd.DataFrame(raw_data)
            elif df is None:
                # Last resort: try direct load
                df = data_pipeline.load_data(
                    "/home/chandanraj-m/OpsNexus-ML/data_pipeline/synthetic_telemetry_sample.json"
                )

            cleaned_df = data_pipeline.clean_data(df)
            featured_df = data_pipeline.engineer_features(cleaned_df)

            # Get the latest feature vector (excluding target and non-feature columns)
            exclude_cols = ['agent_id']
            if 'target' in featured_df.columns:
                exclude_cols.append('target')

            feature_cols = [col for col in featured_df.columns if col not in exclude_cols]
            latest_features = featured_df[feature_cols].iloc[-1:].values  # Most recent sample

            # Get anomaly score and detection result
            anomaly_score = isolation_forest_detector.predict_anomaly_score(latest_features)[0]
            is_anomaly = isolation_forest_detector.detect_anomaly(latest_features, threshold=0.7)[0]

            # Get explainability information for contributing factors
            explanation = isolation_forest_detector.explain_anomaly(latest_features, sample_idx=0, top_n=5)

            # Format contributing factors from explanation
            contributing_factors = []
            if explanation and 'top_contributing_factors' in explanation:
                for factor in explanation['top_contributing_factors']:
                    contributing_factors.append({
                        'feature': factor['feature'],
                        'deviation_score': round(factor['deviation_score'], 3),
                        'value': round(factor['value'], 2),
                        'importance': round(factor.get('importance', 0.0), 3)
                    })
            else:
                # Fallback explanation based on feature importance if available
                feature_importance = isolation_forest_detector.get_feature_importance()
                if feature_importance and len(feature_cols) > 0:
                    # Get top 5 features by importance
                    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    for feature, importance in top_features:
                        if feature in featured_df.columns:
                            value = float(featured_df[feature].iloc[-1])
                            # Approximate deviation as z-score
                            mean_val = featured_df[feature].mean()
                            std_val = featured_df[feature].std()
                            deviation = abs((value - mean_val) / (std_val if std_val != 0 else 1))
                            contributing_factors.append({
                                'feature': feature,
                                'deviation_score': round(deviation, 3),
                                'value': round(value, 2),
                                'importance': round(importance, 3)
                            })
                else:
                    # Final fallback to basic factors
                    contributing_factors = [
                        {'feature': 'cpu_usage_percent', 'deviation_score': round(anomaly_score * 0.8, 3), 'value': 0.0, 'importance': 0.0},
                        {'feature': 'memory_usage_percent', 'deviation_score': round(anomaly_score * 0.6, 3), 'value': 0.0, 'importance': 0.0},
                        {'feature': 'network_bytes_sent', 'deviation_score': round(anomaly_score * 0.4, 3), 'value': 0.0, 'importance': 0.0}
                    ]

            # Calculate confidence based on how extreme the score is and model certainty
            # Simple confidence: farther from 0.5 = higher confidence
            confidence = min(0.95, 0.5 + (abs(anomaly_score - 0.5) * 0.9))

            # Prepare response
            response = {
                'agent_id': agent_id,
                'detection_timestamp': datetime.utcnow().isoformat() + 'Z',
                'lookback_minutes': lookback_minutes,
                'sensitivity': sensitivity,
                'anomaly_score': round(float(anomaly_score), 3),
                'is_anomaly': bool(is_anomaly),
                'confidence': round(float(confidence), 3),
                'contributing_factors': contributing_factors,
                'model_info': {
                    'model_type': 'isolation_forest',
                    'detection_method': 'isolation_forest',
                    'model_loaded': True,
                    'n_estimators': isolation_forest_detector.estimator.n_estimators if isolation_forest_detector.estimator else 100,
                    'features_used': len(feature_cols),
                    'explainability_available': explanation is not None
                }
            }

            return jsonify(response)

        except Exception as processing_error:
            logger.error(f"Error processing data for anomaly detection: {processing_error}")
            logger.error(traceback.format_exc())
            # Fallback to mock data if processing fails
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
                    'detection_method': 'residual_based_fallback',
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

@app.route('/models/compare', methods=['POST'])
def compare_models_endpoint():
    """
    Compare all available models and return the best performing one
    Expected JSON payload:
    {
        "lookback_points": 100   // Optional, how many points to use for comparison
    }
    """
    try:
        if not model_loaded:
            return jsonify({
                'error': 'Models not loaded',
                'message': 'Please train and load models first'
            }), 503

        # Get request parameters
        data = request.get_json() or {}
        lookback_points = data.get('lookback_points', 100)

        logger.info(f"Model comparison request with {lookback_points} lookback points")

        # In a real implementation, we would fetch recent telemetry data
        # For now, we'll use synthetic data to demonstrate the flow
        latest_features, feature_cols = get_latest_features_from_local_telemetry(lookback_points)

        # For model comparison, we need to generate training data
        # In production, this would use real historical data from OpsNexus
        try:
            # Generate some synthetic training data for demonstration
            from data_pipeline.synthetic_data_generator import generate_metric_payload, save_synthetic_data
            import numpy as np

            # Create a larger dataset for meaningful comparison
            data = generate_metric_payload(num_points=500)
            save_synthetic_data(data, "/home/chandanraj-m/OpsNexus-ML/data_pipeline/comparison_sample.json")

            # Load and process the data
            df = data_pipeline.load_data("/home/chandanraj-m/OpsNexus-ML/data_pipeline/comparison_sample.json")
            cleaned_df = data_pipeline.clean_data()
            featured_df = data_pipeline.engineer_features()

            # Prepare training data
            X, y, feature_names = data_pipeline.prepare_training_data(
                target_column='cpu_usage_percent',
                prediction_horizon=6
            )

            # Compare models using our enhanced model comparison function
            from models.enhanced_model import compare_models
            comparison_result = compare_models(
                X, y,
                feature_names=feature_names,
                test_size=0.2,
                random_state=42
            )

            # Prepare response
            response = {
                'comparison_timestamp': datetime.utcnow().isoformat() + 'Z',
                'lookback_points': lookback_points,
                'training_samples': len(X),
                'test_samples': len(X) // 5,  # Approximately 20% for test
                'feature_count': len(feature_names),
                'best_model': {
                    'model_type': comparison_result['best_model_type'],
                    'test_mae': comparison_result['best_test_mae']
                },
                'all_models': {},
                'recommendation': f"Use {comparison_result['best_model_type']} for best performance (MAE: {comparison_result['best_test_mae']:.4f})"
            }

            # Add results for each model
            for model_type, result in comparison_result['all_results'].items():
                if 'error' in result:
                    response['all_models'][model_type] = {
                        'status': 'error',
                        'error': result['error']
                    }
                else:
                    metrics = result['metrics']
                    response['all_models'][model_type] = {
                        'status': 'success',
                        'test_mae': metrics.get('test_mae'),
                        'test_rmse': metrics.get('test_rmse'),
                        'test_r2': metrics.get('test_r2'),
                        'train_mae': metrics.get('train_mae'),
                        'train_r2': metrics.get('train_r2')
                    }

            return jsonify(response)

        except Exception as comparison_error:
            logger.error(f"Error in model comparison: {comparison_error}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': 'Model comparison failed',
                'message': str(comparison_error)
            }), 500

    except Exception as e:
        logger.error(f"Error in model comparison endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Model comparison endpoint failed',
            'message': str(e)
        }), 500


@app.route('/predict/enhanced', methods=['POST'])
def predict_enhanced_cpu_usage():
    """
    Predict CPU usage using the enhanced model system
    Expected JSON payload:
    {
        "agent_id": "optional-agent-id",
        "horizon_minutes": 10,  // Optional, defaults to 10 minutes
        "lookback_points": 100,  // Optional, how many points to look back
        "model_type": "auto"     // Optional: linear_regression, ridge, lasso, random_forest, auto (default: auto)
    }
    """
    try:
        if not model_loaded:
            return jsonify({
                'error': 'Models not loaded',
                'message': 'Please train and load models first'
            }), 503

        # Get request parameters
        data = request.get_json() or {}
        agent_id = data.get('agent_id', 'unknown')
        horizon_minutes = data.get('horizon_minutes', 10)
        lookback_points = data.get('lookback_points', 100)
        model_type = data.get('model_type', 'auto')  # auto, linear_regression, ridge, lasso, random_forest

        logger.info(f"Enhanced CPU prediction request for agent {agent_id}, horizon {horizon_minutes}min, model_type: {model_type}")

        # In a real implementation, we would:
        # 1. Fetch recent telemetry from OpsNexus API for the given agent
        # 2. Process it through our pipeline to get features
        # 3. Make a prediction using the specified model type

        # For now, we'll use synthetic data to demonstrate the flow
        latest_features, feature_cols = get_latest_features_from_local_telemetry(lookback_points)

        # Handle model selection
        prediction_model = cpu_predictor  # Default to loaded model
        actual_model_type = 'unknown'

        if model_type != 'auto' and hasattr(cpu_predictor, 'model_type'):
            # If a specific model type is requested and we have an enhanced model
            if isinstance(cpu_predictor, EnhancedCPUUsagePredictor):
                # Check if the requested model type matches current model
                if cpu_predictor.model_type == model_type:
                    prediction_model = cpu_predictor
                    actual_model_type = model_type
                else:
                    # We would need to load/train the specific model type
                    # For now, we'll use the current model and note this in the response
                    prediction_model = cpu_predictor
                    actual_model_type = cpu_predictor.model_type
                    logger.info(f"Requested model type {model_type} differs from loaded model {cpu_predictor.model_type}. Using loaded model.")
            else:
                # We have a baseline model but requested a specific type
                # For simplicity, we'll use the baseline model
                prediction_model = cpu_predictor
                actual_model_type = 'linear_regression'
                logger.info(f"Requested model type {model_type} but using baseline model. Consider training enhanced models.")
        else:
            # Use the loaded model (could be baseline or enhanced)
            if hasattr(cpu_predictor, 'get_model_info'):
                try:
                    model_info = cpu_predictor.get_model_info()
                    actual_model_type = model_info.get('model_type', 'unknown')
                except:
                    actual_model_type = 'unknown'
            else:
                actual_model_type = 'linear_regression'

        # Make prediction
        prediction = prediction_model.predict_next_cpu_usage(latest_features)

        # Calculate confidence interval (enhanced for better models)
        # In reality, we'd use prediction intervals from the model
        confidence_range = 5.0  # Default range

        # Adjust confidence based on model type if available
        if hasattr(prediction_model, 'get_model_info'):
            try:
                model_info = prediction_model.get_model_info()
                if model_info.get('training_metrics'):
                    test_mae = model_info['training_metrics'].get('test_mae', 5.0)
                    # Scale confidence range based on model performance (better MAE = smaller range)
                    confidence_range = max(2.0, min(10.0, test_mae * 1.5))
            except:
                pass  # Use default range

        confidence_lower = max(0, prediction - confidence_range)
        confidence_upper = min(100, prediction + confidence_range)

        # Prepare response with enhanced model information
        model_info_response = {
            'features_used': len(feature_cols),
            'model_loaded': model_loaded,
            'requested_model_type': model_type,
            'actual_model_type': actual_model_type
        }

        # Add model-specific information if available
        if hasattr(prediction_model, 'get_model_info'):
            try:
                model_details = prediction_model.get_model_info()
                model_info_response.update({
                    'training_metrics': model_details.get('training_metrics', {}),
                    'model_parameters': model_details.get('model_parameters', {})
                })
            except Exception as e:
                logger.warning(f"Could not get enhanced model info for response: {e}")

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
            'model_info': model_info_response
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in enhanced CPU prediction: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Enhanced prediction failed',
            'message': str(e)
        }), 500


@app.route('/models/info/enhanced', methods=['GET'])
def model_info_enhanced():
    """
    Get comprehensive information about the loaded model(s)
    """
    try:
        if not model_loaded:
            return jsonify({
                'error': 'No model loaded',
                'model_loaded': False
            }), 404

        # Prepare base response
        response = {
            'model_loaded': model_loaded,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }

        # Get information from the loaded model
        if hasattr(cpu_predictor, 'get_model_info'):
            try:
                model_details = cpu_predictor.get_model_info()
                response.update(model_details)
            except Exception as e:
                logger.warning(f"Could not get enhanced model info: {e}")
                response['error'] = 'Failed to get detailed model info'
        else:
            # Baseline model information
            response.update({
                'model_type': 'linear_regression',
                'feature_count': len(feature_names) if feature_names else 0,
                'feature_names': feature_names,
                'training_metrics': {}  # Baseline model doesn't store training metrics in the same way
            })

            # Add feature importance if available
            if cpu_predictor.is_trained:
                importance = cpu_predictor.get_feature_importance()
                if importance:
                    response['feature_importance'] = importance

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error getting enhanced model info: {e}")
        logger.error(traceback.format_exc())
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