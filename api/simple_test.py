"""
Simple test for the API functionality without running a server
"""
import sys
import os
import json

# Add project directories to path
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_pipeline'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

from pipeline import OpsNexusDataPipeline
from baseline_model import CPUUsagePredictor

def test_api_logic():
    """Test the core logic that would be used in the API endpoints"""
    print("Testing OpsNexus-ML API Logic...")
    print("=" * 50)

    try:
        # Initialize components (similar to what the API does)
        print("1. Initializing components...")
        data_pipeline = OpsNexusDataPipeline()

        # Load and prepare data
        print("2. Loading and preparing data...")
        data_path = "/home/chandanraj-m/OpsNexus-ML/data_pipeline/synthetic_telemetry_sample.json"
        df = data_pipeline.load_data(data_path)
        cleaned_df = data_pipeline.clean_data()
        featured_df = data_pipeline.engineer_features()

        # Prepare training data
        X, y, feature_names = data_pipeline.prepare_training_data(
            target_column='cpu_usage_percent',
            prediction_horizon=6  # Predict 1 minute ahead
        )
        print(f"   Prepared {X.shape[0]} samples with {X.shape[1]} features")

        # Load or train model
        print("3. Loading/training model...")
        model_path = "/home/chandanraj-m/OpsNexus-ML/models/cpu_predictor.pkl"
        cpu_predictor = CPUUsagePredictor(model_path=model_path)

        if os.path.exists(model_path):
            cpu_predictor.load_model(model_path)
            print(f"   Loaded existing model from {model_path}")
        else:
            print("   Training new model...")
            metrics = cpu_predictor.train(X, y, feature_names=feature_names)
            cpu_predictor.save_model(model_path)
            print(f"   Model trained and saved. Test MAE: {metrics['test_mae']:.4f}")

        model_loaded = cpu_predictor.is_trained
        print(f"   Model loaded: {model_loaded}")

        # Test prediction logic (similar to /predict/cpu endpoint)
        print("\n4. Testing prediction logic...")
        # Get latest features (simulating recent data)
        latest_df = data_pipeline.load_data(data_path)
        latest_cleaned = data_pipeline.clean_data()
        latest_featured = data_pipeline.engineer_features()

        # Get feature columns (same as used in training)
        exclude_cols = ['agent_id']
        if 'target' in latest_featured.columns:
            exclude_cols.append('target')

        feature_cols = [col for col in latest_featured.columns if col not in exclude_cols]
        latest_features = latest_featured[feature_cols].iloc[-1:].values  # Most recent sample

        # Make prediction
        prediction = cpu_predictor.predict_next_cpu_usage(latest_features)

        # Calculate confidence interval (simplified)
        confidence_lower = max(0, prediction - 5.0)
        confidence_upper = min(100, prediction + 5.0)

        print(f"   Predicted CPU usage: {prediction:.2f}%")
        print(f"   Confidence interval: [{confidence_lower:.2f}, {confidence_upper:.2f}]%")
        print(f"   Features used: {len(feature_cols)}")

        # Test anomaly detection logic (similar to /detect/anomaly endpoint)
        print("\n5. Testing anomaly detection logic...")
        # Simple anomaly detection based on prediction error
        # Get actual next value (if we had it) - for demo we'll use a simple heuristic

        # Mock anomaly score based on how unusual the prediction is
        # In reality, this would compare to learned normal patterns
        cpu_mean = latest_featured['cpu_usage_percent'].mean()
        cpu_std = latest_featured['cpu_usage_percent'].std()

        # Z-score of the prediction relative to recent data
        if cpu_std > 0:
            z_score = abs((prediction - cpu_mean) / cpu_std)
            # Convert z-score to anomaly score (0-1 range)
            anomaly_score = min(1.0, z_score / 3.0)  # 3-sigma as max
        else:
            anomaly_score = 0.1

        is_anomaly = anomaly_score > 0.7

        # Mock contributing factors
        contributing_factors = [
            {'metric': 'cpu_usage_percent', 'score': round(anomaly_score * 0.8, 3)},
            {'metric': 'memory_usage_percent', 'score': round(anomaly_score * 0.6, 3)},
            {'metric': 'network_bytes_sent', 'score': round(anomaly_score * 0.4, 3)}
        ]

        print(f"   Anomaly score: {anomaly_score:.3f}")
        print(f"   Is anomaly: {is_anomaly}")
        print(f"   Contributing factors: {len(contributing_factors)}")

        # Test model info logic (similar to /models/info endpoint)
        print("\n6. Testing model info logic...")
        importance = cpu_predictor.get_feature_importance()

        print(f"   Model type: linear_regression")
        print(f"   Feature count: {len(feature_names) if feature_names else 0}")
        if importance:
            top_features = dict(list(sorted(importance.items(),
                                          key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
                                          reverse=True)[:5]))
            print(f"   Top 5 features:")
            for feature, coef in top_features.items():
                print(f"     {feature}: {coef:.4f}")

        print("\n✅ All API logic tests passed!")
        return True

    except Exception as e:
        print(f"❌ API logic test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_api_logic()
    sys.exit(0 if success else 1)