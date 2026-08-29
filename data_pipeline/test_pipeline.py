"""
Test script for the data pipeline
"""
import sys
import os

# Add the data_pipeline directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from pipeline import OpsNexusDataPipeline
import json

def test_pipeline():
    """
    Test the data pipeline functionality
    """
    print("Testing OpsNexus-ML Data Pipeline...")

    # Initialize pipeline
    pipeline = OpsNexusDataPipeline()

    # Load sample data
    data_path = "/home/chandanraj-m/OpsNexus-ML/data_pipeline/synthetic_telemetry_sample.json"

    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        return False

    try:
        # Load data
        print("1. Loading data...")
        df = pipeline.load_data(data_path)
        print(f"   Loaded {len(df)} records")
        print(f"   Columns: {list(df.columns)}")

        # Show data info
        print("2. Getting data info...")
        info = pipeline.get_data_info()
        print(f"   Date range: {info['date_range']['start']} to {info['date_range']['end']}")

        # Clean data
        print("3. Cleaning data...")
        cleaned_df = pipeline.clean_data()
        print(f"   Cleaned shape: {cleaned_df.shape}")

        # Engineer features
        print("4. Engineering features...")
        featured_df = pipeline.engineer_features()
        print(f"   Featured shape: {featured_df.shape}")
        print(f"   Number of features: {len(featured_df.columns)}")

        # Prepare training data
        print("5. Preparing training data...")
        X, y, feature_cols = pipeline.prepare_training_data(
            target_column='cpu_usage_percent',
            prediction_horizon=6  # Predict 1 minute ahead (6 * 10s)
        )
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        print(f"   Feature columns ({len(feature_cols)}): {feature_cols[:5]}...")  # Show first 5

        # Show some sample values
        print("6. Sample data preview:")
        print("   First 5 rows of target variable (next 1min CPU usage):")
        for i in range(min(5, len(y))):
            print(f"     {y[i]:.2f}%")

        print("\n✅ Pipeline test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Pipeline test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)