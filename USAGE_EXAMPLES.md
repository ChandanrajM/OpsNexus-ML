# OpsNexus-ML Usage Examples

This file contains practical examples demonstrating how to use the enhanced features of OpsNexus-ML.

## Table of Contents
1. [Basic Usage](#basic-usage)
2. [Model Comparison](#model-comparison)
3. [Enhanced Predictions](#enhanced-predictions)
4. [Visualization](#visualization)
5. [Integration with OpsNexus](#integration-with-opsnexus)
6. [Advanced Data Pipeline](#advanced-data-pipeline)

## Basic Usage

### Starting the Service
```bash
# Activate virtual environment
source venv/bin/activate

# Start the API
python api/app.py
```

### Making a Basic Prediction Request
```bash
curl -X POST http://localhost:5000/predict/cpu \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "web-server-01",
    "horizon_minutes": 10,
    "lookback_points": 100
  }'
```

### Making an Enhanced Prediction Request
```bash
curl -X POST http://localhost:5000/predict/enhanced \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "web-server-01",
    "horizon_minutes": 10,
    "lookback_points": 100,
    "model_type": "random_forest"
  }'
```

## Model Comparison

### Comparing All Available Models
```bash
curl -X POST http://localhost:5000/models/compare \
  -H "Content-Type: application/json" \
  -d '{
    "lookback_points": 100
  }'
```

### Programmatic Model Comparison
```python
from models.enhanced_model import compare_models
from data_pipeline.pipeline import OpsNexusDataPipeline

# Load and prepare data
pipeline = OpsNexusDataPipeline()
df = pipeline.load_data('data_pipeline/synthetic_telemetry_sample.json')
cleaned = pipeline.clean_data()
featured = pipeline.engineer_features()
X, y, feature_names = pipeline.prepare_training_data(
    target_column='cpu_usage_percent', 
    prediction_horizon=6
)

# Compare models
comparison = compare_models(X, y, feature_names=feature_names)

print(f"Best model: {comparison['best_model_type']}")
print(f"Best test MAE: {comparison['best_test_mae']:.4f}")

# Access individual model results
for model_type, result in comparison['all_results'].items():
    if 'error' not in result:
        metrics = result['metrics']
        print(f"{model_type}: MAE={metrics['test_mae']:.4f}, R²={metrics['test_r2']:.4f}")
```

## Enhanced Predictions

### Using Different Model Types
```python
from models.enhanced_model import EnhancedCPUUsagePredictor, ModelType
from data_pipeline.pipeline import OpsNexusDataPipeline

# Load data
pipeline = OpsNexusDataPipeline()
df = pipeline.load_data('data_pipeline/synthetic_telemetry_sample.json')
cleaned = pipeline.clean_data()
featured = pipeline.engineer_features()
X, y, feature_names = pipeline.prepare_training_data(
    target_column='cpu_usage_percent', 
    prediction_horizon=6
)

# Train different models
models = {
    'linear_regression': ModelType.LINEAR_REGRESSION,
    'ridge': ModelType.RIDGE,
    'lasso': ModelType.LASSO,
    'random_forest': ModelType.RANDOM_FOREST
}

predictions = {}
for name, model_type in models.items():
    predictor = EnhancedCPUUsagePredictor(model_type=model_type)
    predictor.train(X, y, feature_names=feature_names)
    pred = predictor.predict_next_cpu_usage(X[0:1])  # Predict first sample
    predictions[name] = pred
    print(f"{name}: {pred:.2f}%")

# Get feature importance from the best performing model
best_model_type = min(predictions, key=predictions.get)  # Lowest prediction error
best_predictor = EnhancedCPUUsagePredictor(model_type=models[best_model_type])
best_predictor.train(X, y, feature_names=feature_names)
importance = best_predictor.get_feature_importance()

print(f"\nFeature importance from {best_model_type}:")
for feature, imp in sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
    if feature != 'intercept':
        print(f"  {feature}: {imp:.4f}")
```

## Visualization

### Generating All Visualizations
```bash
python -m visualization.enhanced_visualization
```

### Generating Specific Visualizations
```python
from visualization.enhanced_visualization import OpsNexusMLVisualizer

# Initialize visualizer
viz = OpsNexusMLVisualizer()

# Generate model comparison
viz.plot_model_comparison()
print("Model comparison saved!")

# Generate anomaly detection insights
viz.plot_anomaly_detection_insights()
print("Anomaly detection insights saved!")

# Generate training progress dashboard
viz.plot_training_progress_dashboard()
print("Training progress dashboard saved!")

# Generate comprehensive HTML report
viz.create_comprehensive_report()
print("Comprehensive HTML report saved!")
```

### Accessing Generated Visualizations
After running the visualization suite, you'll find:
- `visualization/model_comparison.png`
- `visualization/anomaly_detection_insights.png`
- `visualization/training_progress_dashboard.png`
- `visualization/opsnexus_ml_report.html`

## Integration with OpsNexus

### Example Integration Code for OpsNexus Platform
```python
import requests
import json

class OpsNexusMLClient:
    def __init__(self, ml_service_url="http://opsnexus-ml:5000"):
        self.ml_service_url = ml_service_url
    
    def get_cpu_prediction(self, agent_id, horizon_minutes=10, model_type="auto"):
        """Get CPU usage prediction for capacity planning"""
        response = requests.post(
            f"{self.ml_service_url}/predict/enhanced",
            json={
                "agent_id": agent_id,
                "horizon_minutes": horizon_minutes,
                "model_type": model_type
            }
        )
        return response.json()
    
    def get_anomaly_score(self, agent_id, lookback_minutes=30, sensitivity="medium"):
        """Get anomaly score for alerting"""
        response = requests.post(
            f"{self.ml_service_url}/detect/anomaly",
            json={
                "agent_id": agent_id,
                "lookback_minutes": lookback_minutes,
                "sensitivity": sensitivity
            }
        )
        return response.json()
    
    def compare_models(self, lookback_points=100):
        """Compare all available ML models"""
        response = requests.post(
            f"{self.ml_service_url}/models/compare",
            json={"lookback_points": lookback_points}
        )
        return response.json()

# Usage example
if __name__ == "__main__":
    ml_client = OpsNexusMLClient()
    
    # Get prediction for capacity planning
    prediction = ml_client.get_cpu_prediction(
        agent_id="web-server-01",
        horizon_minutes=60,  # 1-hour ahead prediction
        model_type="random_forest"  # or "auto" for best model
    )
    
    if prediction.get('model_info', {}).get('model_loaded'):
        predicted_cpu = prediction['predicted_cpu_usage_percent']
        confidence = prediction['confidence_interval']
        print(f"Predicted CPU usage: {predicted_cpu:.1f}% "
              f"(range: {confidence['lower']:.1f}-{confidence['upper']:.1f}%)")
        
        # Capacity planning logic
        if predicted_cpu > 80:
            print("⚠️  WARNING: Predicted high CPU usage - consider scaling up")
        elif predicted_cpu > 95:
            print("🚨 CRITICAL: Predicted CPU overload - immediate action required")
    
    # Get anomaly score for alerting
    anomaly_result = ml_client.get_anomaly_score(
        agent_id="web-server-01",
        lookback_minutes=15,
        sensitivity="high"
    )
    
    if anomaly_result.get('is_anomaly'):
        score = anomaly_result['anomaly_score']
        factors = anomaly_result['contributing_factors']
        print(f"🚨 ANOMALY DETECTED: Score {score:.3f}")
        print("Top contributing factors:")
        for factor in factors[:3]:  # Top 3 factors
            print(f"  - {factor['feature']}: {factor['deviation_score']:.3f}")
```

## Advanced Data Pipeline

### Using Caching for Performance
```python
from data_pipeline.pipeline import OpsNexusDataPipeline
import time

# Enable caching for repeated operations
pipeline = OpsNexusDataPipeline(enable_caching=True)

# First load - will read from file
start_time = time.time()
df1 = pipeline.load_data('data_pipeline/synthetic_telemetry_sample.json')
load_time_1 = time.time() - start_time

# Second load - will use cache (much faster)
start_time = time.time()
df2 = pipeline.load_data('data_pipeline/synthetic_telemetry_sample.json')
load_time_2 = time.time() - start_time

print(f"First load: {load_time_1:.3f}s")
print(f"Second load (cached): {load_time_2:.3f}s")
print(f"Speedup: {load_time_1/load_time_2:.1f}x")

# The DataFrames should be identical
assert df1.equals(df2)
```

### Advanced Feature Engineering
```python
from data_pipeline.pipeline import OpsNexusDataPipeline
import pandas as pd

pipeline = OpsNexusDataPipeline()

# Load and process data
df = pipeline.load_data('data_pipeline/synthetic_telemetry_sample.json')
cleaned = pipeline.clean_data()
featured = pipeline.engineer_features()

print("Available features:")
for i, feature in enumerate(sorted(featured.columns), 1):
    print(f"{i:2d}. {feature}")

# Check for advanced features
advanced_features = [
    'cpu_ema_12', 'memory_ema_12',  # Exponential moving averages
    'cpu_momentum', 'memory_momentum',  # Momentum indicators
    'cpu_volatility', 'memory_volatility'  # Volatility indicators
]

print("\nAdvanced features present:")
for feature in advanced_features:
    if feature in featured.columns:
        print(f"  ✓ {feature}")
    else:
        print(f"  ✗ {feature}")

# Use feature scaling
X_scaled, y, feature_names = pipeline.prepare_training_data(
    target_column='cpu_usage_percent',
    prediction_horizon=6,
    scale_features=True,
    feature_scale_type='standard'  # or 'minmax'
)

print(f"\nPrepared {len(X_scaled)} training samples with {len(feature_names)} features")
print(f"Feature scaling applied: standard")

# Get the scaler for consistent transformation (useful for production)
scaler = pipeline.get_feature_scaler(feature_names, scale_type='standard')
if scaler is not None:
    print("Fitted scaler retrieved for production use")
```

### Getting Data Quality Information
```python
from data_pipeline.pipeline import OpsNexusDataPipeline

pipeline = OpsNexusDataPipeline()
df = pipeline.load_data('data_pipeline/synthetic_telemetry_sample.json')

info = pipeline.get_data_info()

print("=== Data Information ===")
print(f"Total records: {info['total_records']}")
print(f"Date range: {info['date_range']['start']} to {info['date_range']['end']}")
print(f"Columns: {len(info['columns'])}")

if 'memory_usage_mb' in info:
    print(f"Memory usage: {info['memory_usage_mb']:.2f} MB")

if 'quality_metrics' in info:
    qm = info['quality_metrics']
    print(f"Complete records: {qm['complete_records']} ({qm['completeness_percentage']:.1f}%)")
    print(f"Duplicate records: {qm['duplicate_records']}")
    print(f"Memory efficiency: {qm['memory_efficiency']}")

# Check for missing values
missing_cols = [col for col, count in info['missing_values'].items() if count > 0]
if missing_cols:
    print(f"Columns with missing values: {missing_cols}")
else:
    print("No missing values detected")
```

## Model Persistence and Versioning

### Saving and Loading Models
```python
from models.enhanced_model import EnhancedCPUUsagePredictor, ModelType
from data_pipeline.pipeline import OpsNexusDataPipeline
import os

# Train a model
pipeline = OpsNexusDataPipeline()
df = pipeline.load_data('data_pipeline/synthetic_telemetry_sample.json')
cleaned = pipeline.clean_data()
featured = pipeline.engineer_features()
X, y, feature_names = pipeline.prepare_training_data(
    target_column='cpu_usage_percent',
    prediction_horizon=6
)

# Train and save an enhanced model
predictor = EnhancedCPUUsagePredictor(model_type=ModelType.RANDOM_FOREST)
metrics = predictor.train(X, y, feature_names=feature_names)

# Save with custom path
model_path = 'models/my_custom_rf_model.pkl'
predictor.save_model(model_path)
print(f"Model saved to {model_path}")

# Load the model later
loaded_predictor = EnhancedCPUUsagePredictor(model_type=ModelType.RANDOM_FOREST, model_path=model_path)
loaded_predictor.load_model(model_path)

# Verify they produce the same results
original_pred = predictor.predict_next_cpu_usage(X[0:1])
loaded_pred = loaded_predictor.predict_next_cpu_usage(X[0:1])

print(f"Original prediction: {original_pred:.4f}")
print(f"Loaded prediction: {loaded_pred:.4f}")
print(f"Match: {abs(original_pred - loaded_pred) < 1e-10}")

# Get model information
info = loaded_predictor.get_model_info()
print(f"\nModel Info:")
print(f"  Type: {info['model_type']}")
print(f"  Trained: {info['is_trained']}")
print(f"  Features: {info['feature_count']}")
if 'training_metrics' in info:
    metrics = info['training_metrics']
    print(f"  Test MAE: {metrics.get('test_mae', 'N/A'):.4f}")
    print(f"  Test R²: {metrics.get('test_r2', 'N/A'):.4f}")
```

## Troubleshooting

### Common Issues and Solutions

1. **"ModuleNotFoundError: No module named 'sklearn'"**
   - Solution: Install scikit-learn: `pip install scikit-learn`

2. **"Model not loaded" error from API**
   - Solution: Train and save a model first using `python models/test_model.py` or the enhanced model training code above

3. **Poor model performance**
   - Solutions:
     - Try different model types using the model comparison endpoint
     - Increase the amount of training data
     - Adjust feature engineering parameters
     - Try different prediction horizons

4. **Slow API response**
   - Solutions:
     - Enable caching in the data pipeline
     - Use simpler models for faster prediction (linear regression)
     - Pre-load models at startup (already done in the API)
     - Consider using a production WSGI server like Gunicorn

### Getting Help
- Check the logs: The API logs detailed information to stdout/stderr
- Run tests: `python -m pytest tests/ -v` to verify functionality
- Read the source code: All functions have detailed docstrings
- Create an issue: Report problems on the GitHub repository