"""
Visualization script for OpsNexus-ML
Creates plots showing actual vs predicted values and model performance
"""
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add project directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_pipeline'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

from pipeline import OpsNexusDataPipeline
from baseline_model import CPUUsagePredictor

def plot_actual_vs_predicted():
    """
    Create a plot showing actual vs predicted CPU usage over time
    """
    print("Creating actual vs predicted CPU usage plot...")

    try:
        # Initialize pipeline and load data
        pipeline = OpsNexusDataPipeline()
        data_path = "/home/chandanraj-m/OpsNexus-ML/data_pipeline/synthetic_telemetry_sample.json"

        if not os.path.exists(data_path):
            print(f"Data file not found: {data_path}")
            return False

        # Load and process data
        df = pipeline.load_data(data_path)
        cleaned_df = pipeline.clean_data()
        featured_df = pipeline.engineer_features()

        # Prepare data for prediction
        X, y, feature_names = pipeline.prepare_training_data(
            target_column='cpu_usage_percent',
            prediction_horizon=6  # Predict 1 minute ahead
        )

        # Load or train model
        model_path = "/home/chandanraj-m/OpsNexus-ML/models/cpu_predictor.pkl"
        model = CPUUsagePredictor(model_path=model_path)

        if os.path.exists(model_path):
            model.load_model(model_path)
            print(f"Loaded model from {model_path}")
        else:
            print("Training new model...")
            metrics = model.train(X, y, feature_names=feature_names)
            model.save_model(model_path)
            print(f"Model trained and saved. Test MAE: {metrics['test_mae']:.4f}")

        # Make predictions
        y_pred = model.predict(X)

        # Create time indices for plotting (since we shifted for prediction)
        # We need to align predictions with the actual future values
        # The target y[i] corresponds to X[i+horizon] actually
        # So we plot y[i] against y_pred[i] where both represent time point i+horizon

        # Get the timestamps for the actual future values
        # The original dataframe has timestamps, we need to shift them back by horizon
        horizon_steps = 6
        original_timestamps = pd.to_datetime(cleaned_df.index)
        # The predictions y_pred[i] correspond to actual values at time i+horizon
        # So we take timestamps from horizon_steps onwards
        plot_timestamps = original_timestamps[horizon_steps:len(y)+horizon_steps]

        # Create the plot
        plt.figure(figsize=(12, 6))

        # Plot actual values
        plt.plot(plot_timestamps, y, label='Actual CPU Usage', alpha=0.8, linewidth=2)
        # Plot predicted values
        plt.plot(plot_timestamps, y_pred, label='Predicted CPU Usage', alpha=0.8, linewidth=2, linestyle='--')

        plt.title('Actual vs Predicted CPU Usage Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('CPU Usage (%)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot
        plot_path = "/home/chandanraj-m/OpsNexus-ML/visualization/actual_vs_predicted.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")

        # Also create a scatter plot showing prediction accuracy
        plt.figure(figsize=(8, 8))
        plt.scatter(y, y_pred, alpha=0.6, s=30)

        # Add perfect prediction line
        min_val = min(min(y), min(y_pred))
        max_val = max(max(y), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        plt.title('Prediction Accuracy: Actual vs Predicted Values', fontsize=14, fontweight='bold')
        plt.xlabel('Actual CPU Usage (%)', fontsize=12)
        plt.ylabel('Predicted CPU Usage (%)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()

        # Save the scatter plot
        scatter_path = "/home/chandanraj-m/OpsNexus-ML/visualization/prediction_accuracy.png"
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy plot saved to: {scatter_path}")

        # Create residuals plot
        residuals = y - y_pred
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(plot_timestamps, residuals, alpha=0.7, linewidth=1)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        plt.title('Prediction Residuals Over Time', fontsize=12)
        plt.xlabel('Time')
        plt.ylabel('Residual (Actual - Predicted)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.8)
        plt.title('Distribution of Residuals', fontsize=12)
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the residuals plot
        residuals_path = "/home/chandanraj-m/OpsNexus-ML/visualization/residuals_analysis.png"
        plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
        print(f"Residuals plot saved to: {residuals_path}")

        # Print some statistics
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        r2 = 1 - (np.sum(residuals**2) / np.sum((y - np.mean(y))**2))

        print(f"\nPrediction Statistics:")
        print(f"  Mean Absolute Error (MAE): {mae:.4f}")
        print(f"  Root Mean Square Error (RMSE): {rmse:.4f}")
        print(f"  R² Score: {r2:.4f}")

        return True

    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

def plot_feature_importance():
    """
    Create a bar chart showing feature importance from the trained model
    """
    print("Creating feature importance plot...")

    try:
        # Load model
        model_path = "/home/chandanraj-m/OpsNexus-ML/models/cpu_predictor.pkl"
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False

        model = CPUUsagePredictor(model_path=model_path)
        model.load_model(model_path)

        # Get feature importance
        importance = model.get_feature_importance()
        if not importance:
            print("No feature importance available")
            return False

        # Remove intercept for clarity in this plot
        feature_importance = {k: v for k, v in importance.items() if k != 'intercept'}

        # Sort by absolute value
        sorted_features = sorted(feature_importance.items(),
                               key=lambda x: abs(x[1]),
                               reverse=True)

        # Take top 15 features for readability
        top_features = sorted_features[:15]
        features, values = zip(*top_features)

        # Create horizontal bar chart
        plt.figure(figsize=(10, 8))
        y_pos = np.arange(len(features))
        colors = ['red' if v < 0 else 'blue' for v in values]
        plt.barh(y_pos, values, color=colors, alpha=0.7)
        plt.yticks(y_pos, features)
        plt.xlabel('Coefficient Value', fontsize=12)
        plt.title('Top 15 Feature Importance (Linear Regression Coefficients)',
                  fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()

        # Save the plot
        plot_path = "/home/chandanraj-m/OpsNexus-ML/visualization/feature_importance.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {plot_path}")

        return True

    except Exception as e:
        print(f"Error creating feature importance plot: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to create all visualizations
    """
    print("OpsNexus-ML Visualization Suite")
    print("=" * 40)

    # Create visualization directory
    viz_dir = "/home/chandanraj-m/OpsNexus-ML/visualization"
    os.makedirs(viz_dir, exist_ok=True)

    success1 = plot_actual_vs_predicted()
    print()
    success2 = plot_feature_importance()

    if success1 and success2:
        print("\n✅ All visualizations created successfully!")
        print(f"Plots saved in: {viz_dir}")
        return True
    else:
        print("\n❌ Some visualizations failed to create")
        return False

if __name__ == "__main__":
    # Import pandas here to avoid issues if not installed
    global pd
    try:
        import pandas as pd
    except ImportError:
        print("Please install pandas: pip install pandas")
        sys.exit(1)

    success = main()
    sys.exit(0 if success else 1)