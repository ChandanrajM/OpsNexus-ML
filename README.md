# OpsNexus-ML: Machine Learning Intelligence Layer for OpsNexus

OpsNexus-ML is a machine learning service designed to add intelligent capabilities to the OpsNexus monitoring platform. It provides resource usage prediction and anomaly detection for system telemetry data.

## 🚀 Enhanced Features

This enhanced version includes significant improvements over the original implementation:

### 🔬 Model Enhancements
- **Multiple ML Algorithms**: Linear Regression, Ridge, Lasso, and Random Forest regressors
- **Automatic Model Selection**: Compare and select the best performing model automatically
- **Enhanced Model Persistence**: Improved model saving/loading with metadata
- **Feature Importance Extraction**: Consistent interface across all model types
- **Cross-Validation Support**: Better performance estimation during training

### 📊 Visualization Improvements
- **Model Comparison Dashboard**: Side-by-side comparison of all algorithms
- **Anomaly Detection Insights**: Detailed visualizations of anomaly scores and contributing factors
- **Training Progress Dashboard**: Learning curves, feature distributions, and performance metrics
- **Comprehensive HTML Reports**: Single-file reports with all visualizations
- **Enhanced Plotting**: Better styling, color schemes, and information density

### 🔌 API Extensions
- **Model Comparison Endpoint**: `/models/compare` - Compare all available models
- **Enhanced Prediction Endpoint**: `/predict/enhanced` - Predict using specific model types
- **Enhanced Model Info Endpoint**: `/models/info/enhanced` - Detailed model information
- **Improved Responses**: Richer metadata including training metrics and model parameters
- **Backward Compatibility**: Existing endpoints continue to work unchanged

### ⚡ Data Pipeline Optimization
- **Enhanced Error Handling**: Robust validation and error recovery
- **Performance Caching**: Optional caching for repeated operations
- **Advanced Feature Engineering**: EMA, momentum, volatility indicators
- **Feature Scaling Options**: StandardScaler and MinMaxScaler support
- **Improved Data Validation**: Better handling of malformed input data
- **Data Quality Metrics**: Comprehensive statistics about loaded data

### 🧪 Testing & Evaluation
- **Comprehensive Test Suite**: Unit tests for all major components
- **Enhanced Test Coverage**: Edge cases and error conditions
- **Model Validation Procedures**: Standardized validation protocols
- **Benchmarking Capabilities**: Performance comparison tools

## 📋 Table of Contents
- [Overview](#overview)
- [Enhanced Features](#-enhanced-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Training](#model-training)
- [Development](#development)
- [Docker Deployment](#docker-deployment)
- [Future Integration](#future-integration)
- [Enhanced Visualization](#-enhanced-visualization)
- [Testing](#testing)
- [Documentation](#documentation)

## 🔍 Overview

OpsNexus-ML is designed as a standalone microservice that consumes telemetry data from the existing OpsNexus platform and provides machine learning insights back to it. The service focuses on two core capabilities:

1. **Resource Prediction** - Forecast future system resource usage (CPU, memory, etc.)
2. **Anomaly Detection** - Identify unusual system behavior compared to learned normal patterns

This service does not modify the existing OpsNexus codebase but instead integrates via well-defined APIs.

## 🏗️ Architecture

```
┌─────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│   Linux Hosts       │    │   OpsNexus Backend   │    │   OpsNexus-ML API    │
│ (Servers/Containers)│    │  (Existing System)   │    │  (This Service)      │
└─────────────────────┘    └──────────────────────┘    └──────────────────────┘
          │                         │                         │
          │ HTTP POST /telemetry    │                         │
          │ (every 10s)             │                         │
          ▼                         ▼                         ▼
┌─────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│  OpsNexus Agents    │    │  Time-Series DB      │    │  Data Pipeline       │
│  (Collect metrics)  │    │  (PostgreSQL)        │    │  (Ingest → Clean →   │
└─────────────────────┘    └──────────────────────┘    │   Feature Eng)       │
          │                         │                   └──────────────────────┘
          │                         │                           │
          │                         │ GET /api/v1/analytics   │
          │                         │ (Historical telemetry)  │
          │                         ▼                           ▼
          │                   ┌─────────────────────┐    ┌──────────────────────┐
          │                   │    ML Models        │    │  Inference API       │
          │                   │  (Training/Serving) │    │  (/predict,          │
          │                   │                     │    │   /detect-anomaly)   │
          │                   └─────────────────────┘    └──────────────────────┘
          │                         │                           │
          │                         │                           ▼
          │                         │                   ┌──────────────────────┐
          │                         │                   │  OpsNexus Dashboard  │
          │                         │                   │  (Future Enhancement)│
          │                         │                   │  - Prediction        │
          │                         │                   │    overlays          │
          │                         │                   │  - Anomaly           │
          │                         │                   │    highlights        │
          │                         │                   │  - ML-enhanced       │
          │                         │                   │    alerting          │
          │                         │                   └──────────────────────┘
          │                         │
          │                         ▼
          │                   ┌─────────────────────┐
          │                   │  Alerting System    │
          │                   │  (Future Enhancement)│
          │                   │  - ML-based alerts  │
          │                   │  - Anomaly-triggered notifications
          │                   │  - Correlation with existing alert rules
          │                   └─────────────────────┘
```

### Key Design Principles
- **Zero modifications** to existing OpsNexus code
- **API-based integration** using existing `/analytics` endpoint
- **Technology flexibility** (Python/ML ecosystem vs OpsNexus Go)
- **Independent deployment** and scaling
- **Failure isolation** - service degradation doesn't affect core monitoring
- **Enhanced extensibility** for adding new models and features

## 📁 Project Structure

```
OpsNexus-ML/
├── api/                 # REST API service
│   ├── app.py          # Flask API endpoints (enhanced)
│   ├── test_api.py     # API tests
│   └── simple_test.py  # Logic tests without server
├── data_pipeline/       # Enhanced data ingestion and processing
│   ├── pipeline.py     # Main data processing logic (optimized)
│   ├── synthetic_data_generator.py  # Test data generation
│   └── test_pipeline.py # Pipeline tests
├── models/              # Machine learning models
│   ├── baseline_model.py       # Linear regression predictor
│   ├── enhanced_model.py       # Multiple algorithms with auto-selection
│   ├── isolation_forest_detector.py  # Anomaly detection
│   ├── test_model.py           # Model tests
│   ├── test_enhanced_model.py  # Enhanced model tests
│   ├── cpu_predictor.pkl       # Saved baseline model (generated)
│   ├── enhanced_cpu_predictor.pkl  # Saved enhanced model (generated)
│   └── isolation_forest_detector.pkl  # Saved anomaly detector (generated)
├── config/              # Configuration files
├── docs/                # Documentation
├── visualization/       # Enhanced plotting and visualization
│   ├── plot_predictions.py     # Original visualization
│   ├── enhanced_visualization.py # Enhanced visualization suite
│   └── __init__.py
├── requirements.txt     # Python dependencies (updated)
├── Dockerfile           # Containerization
├── docker-compose.yml   # Docker compose configuration
└── README.md            # This file
```

## ⚙️ Installation

### Prerequisites
- Python 3.12+
- pip (Python package manager)
- Git (for cloning)

### Local Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd OpsNexus-ML
```

2. **Create and activate virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Generate synthetic test data** (optional but recommended for testing)
```bash
cd data_pipeline
python synthetic_data_generator.py
```

5. **Train the initial model**
```bash
cd ..
python models/test_model.py
```

## 🚀 Usage

### Starting the API Service

```bash
# Activate virtual environment if not already activated
source venv/bin/activate

# Start the Flask API server
python api/app.py
```

The service will be available at `http://localhost:5000`

### Using Docker

```bash
docker-compose up --build
```

The service will be available at `http://localhost:5000`

## 🔌 API Endpoints

### Health Check
```
GET /health
```
Returns service status and model loading state.

**Response:**
```json
{
  "status": "healthy",
  "service": "OpsNexus-ML API",
  "model_loaded": true,
  "timestamp": "2026-09-04T20:00:00Z"
}
```

### CPU Usage Prediction (Original)
```
POST /predict/cpu
```

**Request Body:**
```json
{
  "agent_id": "web-server-01",
  "horizon_minutes": 10,
  "lookback_points": 100
}
```

**Response:**
```json
{
  "agent_id": "web-server-01",
  "prediction_timestamp": "2026-09-04T20:00:00Z",
  "horizon_minutes": 10,
  "predicted_cpu_usage_percent": 65.42,
  "confidence_interval": {
    "lower": 60.42,
    "upper": 70.42
  },
  "model_info": {
    "model_type": "linear_regression",
    "features_used": 51,
    "model_loaded": true
  }
}
```

### Enhanced CPU Usage Prediction (NEW)
```
POST /predict/enhanced
```

**Request Body:**
```json
{
  "agent_id": "web-server-01",
  "horizon_minutes": 10,
  "lookback_points": 100,
  "model_type": "auto"  // linear_regression, ridge, lasso, random_forest, auto (default)
}
```

**Response:**
```json
{
  "agent_id": "web-server-01",
  "prediction_timestamp": "2026-09-04T20:00:00Z",
  "horizon_minutes": 10,
  "predicted_cpu_usage_percent": 65.42,
  "confidence_interval": {
    "lower": 60.42,
    "upper": 70.42
  },
  "model_info": {
    "features_used": 57,
    "model_loaded": true,
    "requested_model_type": "auto",
    "actual_model_type": "random_forest",
    "training_metrics": {
      "model_type": "random_forest",
      "test_mae": 3.21,
      "test_rmse": 4.85,
      "test_r2": 0.892,
      "train_samples": 75,
      "test_samples": 19
    },
    "model_parameters": {
      "n_estimators": 100,
      "max_depth": 10,
      "random_state": 42
    }
  }
}
```

### Model Comparison (NEW)
```
POST /models/compare
```

**Request Body:**
```json
{
  "lookback_points": 100
}
```

**Response:**
```json
{
  "comparison_timestamp": "2026-09-04T20:00:00Z",
  "lookback_points": 100,
  "training_samples": 80,
  "test_samples": 20,
  "feature_count": 57,
  "best_model": {
    "model_type": "random_forest",
    "test_mae": 3.21
  },
  "all_models": {
    "linear_regression": {
      "status": "success",
      "test_mae": 4.85,
      "test_rmse": 6.23,
      "test_r2": 0.721,
      "train_mae": 4.21,
      "train_r2": 0.785
    },
    "ridge": {
      "status": "success",
      "test_mae": 4.72,
      "test_rmse": 6.11,
      "test_r2": 0.738,
      "train_mae": 4.15,
      "train_r2": 0.792
    },
    "lasso": {
      "status": "success",
      "test_mae": 4.91,
      "test_rmse": 6.34,
      "test_r2": 0.708,
      "train_mae": 4.28,
      "train_r2": 0.772
    },
    "random_forest": {
      "status": "success",
      "test_mae": 3.21,
      "test_rmse": 4.85,
      "test_r2": 0.892,
      "train_mae": 2.87,
      "train_r2": 0.915
    }
  },
  "recommendation": "Use random_forest for best performance (MAE: 3.21)"
}
```

### Enhanced Model Information (NEW)
```
GET /models/info/enhanced
```

**Response:**
```json
{
  "model_loaded": true,
  "timestamp": "2026-09-04T20:00:00Z",
  "model_type": "random_forest",
  "is_trained": true,
  "feature_count": 57,
  "feature_names": ["cpu_lag_1", "cpu_lag_2", "memory_lag_1", ...],
  "training_metrics": {
    "model_type": "random_forest",
    "test_mae": 3.21,
    "test_rmse": 4.85,
    "test_r2": 0.892,
    "train_samples": 75,
    "test_samples": 19,
    "n_estimators": 100,
    "contamination": 0.1
  },
  "model_parameters": {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
  },
  "last_loaded": "2026-09-04T20:00:00Z"
}
```

### Anomaly Detection
```
POST /detect/anomaly
```

**Request Body:**
```json
{
  "agent_id": "web-server-01",
  "lookback_minutes": 30,
  "sensitivity": "medium"
}
```

**Response:**
```json
{
  "agent_id": "web-server-01",
  "detection_timestamp": "2026-09-04T20:00:00Z",
  "lookback_minutes": 30,
  "sensitivity": "medium",
  "anomaly_score": 0.234,
  "is_anomaly": false,
  "confidence": 0.756,
  "contributing_factors": [
    {"feature": "cpu_usage_percent", "deviation_score": 0.187, "value": 65.2, "importance": 0.234},
    {"feature": "memory_usage_percent", "deviation_score": 0.140, "value": 45.8, "importance": 0.187},
    {"feature": "network_bytes_sent", "deviation_score": 0.094, "value": 1024000, "importance": 0.156}
  ],
  "model_info": {
    "model_type": "isolation_forest",
    "detection_method": "isolation_forest",
    "model_loaded": true,
    "n_estimators": 100,
    "features_used": 57,
    "explainability_available": true
  }
}
```

### Model Information (Original)
```
GET /models/info
```

## 📊 Model Training

The service supports multiple algorithms for CPU usage prediction. To train or retrain models:

### Using Enhanced Models (Recommended)
```bash
# Train and save the best performing model automatically
python -c "
from models.enhanced_model import compare_models
from data_pipeline.pipeline import OpsNexusDataPipeline
import numpy as np

# Load and prepare data
pipeline = OpsNexusDataPipeline()
df = pipeline.load_data('data_pipeline/synthetic_telemetry_sample.json')
cleaned = pipeline.clean_data()
featured = pipeline.engineer_features()
X, y, feature_names = pipeline.prepare_training_data(target_column='cpu_usage_percent', prediction_horizon=6)

# Compare models and get the best one
comparison = compare_models(X, y, feature_names=feature_names)
best_model = comparison['best_model']
best_model_type = comparison['best_model_type']

# Save the best model
best_model.save_model('models/enhanced_cpu_predictor.pkl')
print(f'Best model: {best_model_type} with Test MAE: {comparison[\"best_test_mae\"]:.4f}')
"
```

### Using Specific Model Types
```bash
# Train a specific model type (e.g., Ridge regression)
python -c "
from models.enhanced_model import EnhancedCPUUsagePredictor, ModelType
from data_pipeline.pipeline import OpsNexusDataPipeline

# Load and prepare data
pipeline = OpsNexusDataPipeline()
df = pipeline.load_data('data_pipeline/synthetic_telemetry_sample.json')
cleaned = pipeline.clean_data()
featured = pipeline.engineer_features()
X, y, feature_names = pipeline.prepare_training_data(target_column='cpu_usage_percent', prediction_horizon=6)

# Train and save Ridge model
predictor = EnhancedCPUUsagePredictor(model_type=ModelType.RIDGE)
metrics = predictor.train(X, y, feature_names=feature_names)
predictor.save_model('models/ridge_cpu_predictor.pkl')
print(f'Ridge model trained. Test MAE: {metrics[\"test_mae\"]:.4f}')
"
```

### Traditional Baseline Model (Original Method)
```bash
# Using the test script (which trains and saves a model)
python models/test_model.py

# Or manually:
python -c "
from data_pipeline.pipeline import OpsNexusDataPipeline
from models.baseline_model import CPUUsagePredictor
import os

# Load data
pipeline = OpsNexusDataPipeline()
df = pipeline.load_data('data_pipeline/synthetic_telemetry_sample.json')
cleaned = pipeline.clean_data()
featured = pipeline.engineer_features()
X, y, features = pipeline.prepare_training_data(target_column='cpu_usage_percent')

# Train model
model = CPUUsagePredictor()
metrics = model.train(X, y, feature_names=features)
model.save_model('models/cpu_predictor.pkl')
print(f'Model trained. Test MAE: {metrics[\"test_mae\"]:.4f}')
"
```

## 🤖 Model Training Details

### Available Model Types
1. **Linear Regression** (`linear_regression`) - Baseline interpretable model
2. **Ridge Regression** (`ridge`) - Regularized linear model (L2 regularization)
3. **Lasso Regression** (`lasso`) - Sparse linear model (L1 regularization)
4. **Random Forest** (`random_forest`) - Ensemble tree-based model

### Feature Engineering
The data pipeline automatically creates features including:
- **Time-based features**: Hour of day, day of week, cyclical encodings
- **Lagged features**: Historical values for predicting future states
- **Rolling statistics**: Moving averages and standard deviations
- **Rate of change**: Derivatives and percentage changes
- **Interaction features**: CPU-memory interactions, load indicators
- **Advanced features**: Exponential moving averages, momentum, volatility indicators

### Model Persistence
All models support saving and loading with metadata:
- Model type and configuration
- Feature names used during training
- Training metrics (MAE, RMSE, R², etc.)
- Model-specific parameters
- Timestamp of last training/loading

## 📈 Enhanced Visualization

Run the enhanced visualization suite to generate comprehensive insights:

```bash
# Generate all visualizations and reports
python -m visualization.enhanced_visualization
```

This will create:
- `model_comparison.png`: Side-by-side model performance analysis
- `anomaly_detection_insights.png`: Detailed anomaly detection visualizations
- `training_progress_dashboard.png`: Model learning and performance diagnostics
- `opsnexus_ml_report.html`: Comprehensive HTML report with all visualizations

### Individual Visualization Components
```bash
# Generate only model comparison
python -c "
from visualization.enhanced_visualization import OpsNexusMLVisualizer
viz = OpsNexusMLVisualizer()
viz.plot_model_comparison()
"

# Generate only anomaly detection insights
python -c "
from visualization.enhanced_visualization import OpsNexusMLVisualizer
viz = OpsNexusMLVisualizer()
viz.plot_anomaly_detection_insights()
"

# Generate only training progress dashboard
python -c "
from visualization.enhanced_visualization import OpsNexusMLVisualizer
viz = OpsNexusMLVisualizer()
viz.plot_training_progress_dashboard()
"
```

## 🐳 Docker Deployment

### Build and Run
```bash
docker-compose up --build
```

### Access the Service
- API: `http://localhost:5000`
- Health check: `http://localhost:5000/health`
- API documentation: Interactive exploration via the endpoints above

### Configuration
Edit `docker-compose.yml` to:
- Change port mappings
- Adjust environment variables
- Modify volume mounts for persistent storage

## 🔮 Future Integration with OpsNexus

Once deployed, OpsNexus can consume OpsNexus-ML insights through:

### 1. **Dashboard Enhancements**
- Add "Predicted CPU" panels alongside actual usage
- Show anomaly scores as system health indicators
- Color-code metrics based on ML-derived insights

### 2. **Alerting System Enhancements**
- Create ML-based alert types (e.g., "Predicted Resource Exhaustion")
- Anomaly-triggered notifications
- Correlation of ML insights with existing alert rules

### 3. **Telemetry Enrichment**
- Add ML prediction fields to telemetry stream
- Include anomaly scores in metric payloads
- Provide model metadata for diagnostic purposes

### Integration Approach
OpsNexus would call the OpsNexus-ML API:
```bash
# Get predictions for capacity planning
POST http://opsnexus-ml:5000/predict/enhanced
{
  "agent_id": "web-server-01",
  "horizon_minutes": 60,
  "model_type": "random_forest"
}

# Get anomaly scores for alerting
POST http://opsnexus-ml:5000/detect/anomaly
{
  "agent_id": "web-server-01",
  "lookback_minutes": 15,
  "sensitivity": "high"
}
```

## 🧪 Testing

Run the test suite to validate functionality:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/unit/test_enhanced_model.py -v
python -m pytest tests/unit/test_data_pipeline.py -v
python -m pytest tests/unit/test_baseline_model.py -v
python -m pytest tests/unit/test_anomaly_detection.py -v
```

## 📚 Documentation

### Key Documentation Files
- `README.md` - This file
- `docs/` - Additional documentation
- `model_versioning.py` - Model versioning system documentation
- Code docstrings - Inline documentation for all modules

### Usage Examples
- `api/simple_test.py` - Logic tests without running the server
- `data_pipeline/test_pipeline.py` - Pipeline functionality tests
- `models/test_model.py` - Baseline model usage example
- `visualization/plot_predictions.py` - Original visualization example
- `visualization/enhanced_visualization.py` - Enhanced visualization suite

## 📈 Performance Benchmarks

Typical performance improvements with enhanced models:

| Model Type | Test MAE | Test RMSE | Test R² | Training Time |
|------------|----------|-----------|---------|---------------|
| Linear Regression | 4.85 | 6.23 | 0.721 | Fast |
| Ridge | 4.72 | 6.11 | 0.738 | Fast |
| Lasso | 4.91 | 6.34 | 0.708 | Fast |
| Random Forest | **3.21** | **4.85** | **0.892** | Moderate |

*Note: Values are illustrative based on synthetic data*

## 📊 College Project Progress

### Phase 1: Foundation ✅
- [x] Synthetic data generator matching OpsNexus MetricPayload
- [x] Basic data pipeline (ingest → clean → feature engineer)
- [x] Simple baseline model (linear regression for CPU prediction)

### Phase 2: Core ML ✅
- [x] Multi-algorithm prediction framework (Linear, Ridge, Lasso, Random Forest)
- [x] Anomaly detection (Isolation Forest - implemented and enhanced)
- [x] Inference API endpoints (/predict, /detect-anomaly, /predict/enhanced, /models/compare)
- [x] Enhanced visualization (actual vs predicted plots, feature importance, model comparison)

### Phase 3: Integration Readiness ✅
- [x] API client for OpsNexus `/analytics` endpoint (configurable with fallbacks)
- [x] Model versioning and persistence
- [x] Comprehensive evaluation metrics (MAE, RMSE, R²)
- [x] Deployment steps documented
- [x] OpsNexus integration guide (enhanced with new features)

## 📝 Implementation Notes

### Design Decisions
1. **Multiple Model Support**: Started with scikit-learn algorithms for robustness and performance
2. **Automatic Model Selection**: Added model comparison functionality for optimal performance
3. **Enhanced Feature Engineering**: Added EMA, momentum, and volatility indicators for better temporal pattern recognition
4. **API Extensions**: Maintained backward compatibility while adding powerful new endpoints
5. **Visualization Suite**: Created comprehensive visualization tools for model insights and debugging
6. **Testing Focus**: Emphasized unit testing for reliability and maintainability
7. **Performance Optimization**: Added caching and efficient data processing pipelines

### Limitations & Future Work
1. **Online Learning**: Planning to implement incremental learning capabilities
2. **Model Registry**: Will implement proper model versioning and A/B testing framework
3. **Deep Learning**: Exploration of LSTM/GRU models for temporal dependencies
4. **Feature Automation**: Automated feature selection and importance ranking
5. **Real-time Streaming**: Integration with Apache Kafka or similar for real-time processing
6. **Model Explainability**: Integration of SHAP values for better interpretability

## 🙋‍♂️ Support

For questions, issues, or contributions:
1. Check the existing documentation
2. Review the source code comments
3. Create an issue in the repository
4. Submit pull requests for enhancements

---

**OpsNexus-ML**: Transforming system monitoring from observation to intelligent anticipation.