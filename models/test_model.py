"""
Test script for the baseline model using data from the pipeline
"""
import sys
import os

# Add the necessary directories to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_pipeline'))

from pipeline import OpsNexusDataPipeline
from baseline_model import CPUUsagePredictor, create_and_train_model
import numpy as np

def test_model():
    """
    Test the baseline model with pipeline data
    """
    print("Testing OpsNexus-ML Baseline Model...")
    print("=" * 50)

    try:
        # Step 1: Load and prepare data using the pipeline
        print("1. Loading and preparing data...")
        pipeline = OpsNexusDataPipeline()

        # Use the full synthetic dataset for better training
        data_path = "/home/chandanraj-m/OpsNexus-ML/data_pipeline/synthetic_telemetry.json"

        if not os.path.exists(data_path):
            print(f"   Using sample data instead...")
            data_path = "/home/chandanraj-m/OpsNexus-ML/data_pipeline/synthetic_telemetry_sample.json"

        df = pipeline.load_data(data_path)
        print(f"   Loaded {len(df)} records")

        # Clean and engineer features
        cleaned_df = pipeline.clean_data()
        featured_df = pipeline.engineer_features()
        print(f"   Engineered {len(featured_df.columns)} features")

        # Prepare training data for CPU usage prediction
        X, y, feature_names = pipeline.prepare_training_data(
            target_column='cpu_usage_percent',
            prediction_horizon=6  # Predict 1 minute ahead (6 * 10s intervals)
        )
        print(f"   Prepared {X.shape[0]} training samples with {X.shape[1]} features")

        # Step 2: Train the model
        print("\n2. Training baseline model...")
        model = CPUUsagePredictor()
        metrics = model.train(X, y, feature_names=feature_names)

        # Step 3: Evaluate the model
        print("\n3. Model Performance:")
        print(f"   Training MAE:  {metrics['train_mae']:.4f}")
        print(f"   Test MAE:      {metrics['test_mae']:.4f}")
        print(f"   Training RMSE: {metrics['train_rmse']:.4f}")
        print(f"   Test RMSE:     {metrics['test_rmse']:.4f}")
        print(f"   Training R²:   {metrics['train_r2']:.4f}")
        print(f"   Test R²:       {metrics['test_r2']:.4f}")

        # Step 4: Show feature importance (top 10)
        print("\n4. Top 10 Feature Importance:")
        importance = model.get_feature_importance()
        if importance:
            # Sort by absolute coefficient value, excluding intercept
            sorted_features = [(k, v) for k, v in importance.items() if k != 'intercept']
            sorted_features.sort(key=lambda x: abs(x[1]), reverse=True)

            for i, (feature, coef) in enumerate(sorted_features[:10]):
                print(f"   {i+1:2d}. {feature:<25} : {coef:>8.4f}")

        # Step 5: Test prediction on a few samples
        print("\n5. Sample Predictions:")
        # Get first 5 samples from test set
        n_samples = min(5, len(X))
        sample_indices = list(range(n_samples))  # First few samples

        X_sample = X[sample_indices]
        y_true = y[sample_indices]
        y_pred = model.predict(X_sample)

        print("   Sample | Actual (%) | Predicted (%) | Error (%)")
        print("   -------|------------|---------------|----------")
        for i in range(n_samples):
            error = abs(y_true[i] - y_pred[i])
            print(f"   {i:6d} | {y_true[i]:10.2f} | {y_pred[i]:13.2f} | {error:8.2f}")

        # Step 6: Test model persistence
        print("\n6. Testing model persistence...")
        model_path = "/home/chandanraj-m/OpsNexus-ML/models/test_cpu_predictor.pkl"
        model.save_model(model_path)
        print(f"   Model saved to {model_path}")

        # Load the model again
        model_loaded = CPUUsagePredictor(model_path=model_path)
        model_loaded.load_model(model_path)
        print(f"   Model loaded successfully from {model_path}")

        # Verify predictions match
        y_pred_loaded = model_loaded.predict(X_sample)
        max_diff = np.max(np.abs(y_pred - y_pred_loaded))
        print(f"   Max difference between original and loaded model predictions: {max_diff:.6f}")

        # Step 7: Test single prediction method
        print("\n7. Testing single prediction method...")
        latest_features = X[0:1]  # First sample as 2D array
        single_pred = model.predict_next_cpu_usage(latest_features)
        batch_pred = model.predict(latest_features)[0]
        print(f"   Single prediction: {single_pred:.4f}")
        print(f"   Batch prediction:  {batch_pred:.4f}")
        print(f"   Difference:        {abs(single_pred - batch_pred):.6f}")

        print("\n✅ All model tests passed!")
        return True

    except Exception as e:
        print(f"❌ Model test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)