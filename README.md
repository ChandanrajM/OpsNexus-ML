# OpsNexus-ML: Machine Learning Intelligence Layer for OpsNexus

OpsNexus-ML is a machine learning service designed to add intelligent capabilities to the OpsNexus monitoring platform. It provides resource usage prediction and anomaly detection for system telemetry data.

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Training](#model-training)
- [Development](#development)
- [Docker Deployment](#docker-deployment)
- [Future Integration](#future-integration)

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
          │                   └─────────────────────┘
```

### Key Design Principles
- **Zero modifications** to existing OpsNexus code
- **API-based integration** using existing `/analytics` endpoint
- **Technology flexibility** (Python/ML ecosystem vs OpsNexus Go)
- **Independent deployment** and scaling
- **Failure isolation** - service degradation doesn't affect core monitoring

## 📁 Project Structure

```
OpsNexus-ML/
├── api/                 # REST API service
│   ├── app.py          # Flask API endpoints
│   ├── test_api.py     # API tests
│   └── simple_test.py  # Logic tests without server
├── data_pipeline/       # Data ingestion and processing
│   ├── pipeline.py     # Main data processing logic
│   ├── synthetic_data_generator.py  # Test data generation
│   └── test_pipeline.py # Pipeline tests
├── models/              # Machine learning models
│   ├── baseline_model.py   # Linear regression predictor
│   ├── test_model.py       # Model tests
│   └── cpu_predictor.pkl   # Saved model (generated)
├── config/              # Configuration files
├── docs/                # Documentation
├── visualization/       # Plotting and visualization code
├── requirements.txt     # Python dependencies
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
  "timestamp": "2026-08-29T20:00:00Z"
}
```

### CPU Usage Prediction
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
  "prediction_timestamp": "2026-08-29T20:00:00Z",
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
  "detection_timestamp": "2026-08-29T20:00:00Z",
  "lookback_minutes": 30,
  "sensitivity": "medium",
  "anomaly_score": 0.234,
  "is_anomaly": false,
  "contributing_factors": [
    {"metric": "cpu_usage_percent", "score": 0.187},
    {"metric": "memory_usage_percent", "score": 0.140},
    {"metric": "network_bytes_sent", "score": 0.094}
  ],
  "model_info": {
    "detection_method": "residual_based",
    "model_loaded": true
  }
}
```

### Model Information
```
GET /models/info
```

**Response:**
```json
{
  "model_loaded": true,
  "model_type": "linear_regression",
  "feature_count": 51,
  "features": ["cpu_lag_1", "cpu_lag_2", "memory_lag_1", ...],
  "top_features": {
    "cpu_rolling_mean_3": 2.9434,
    "is_weekend": -2.8927,
    "memory_rolling_mean_3": -1.9634
  },
  "intercept": 45.231,
  "last_loaded": "2026-08-29T20:00:00Z"
}
```

## 🤖 Model Training

The service uses a linear regression model for CPU usage prediction. To train or retrain the model:

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
POST http://opsnexus-ml:5000/predict/cpu
{
  "agent_id": "web-server-01",
  "horizon_minutes": 60
}

# Get anomaly scores for alerting
POST http://opsnexus-ml:5000/detect/anomaly
{
  "agent_id": "web-server-01",
  "lookback_minutes": 15,
  "sensitivity": "high"
}
```

## 📊 College Project Progress

### Phase 1: Foundation ✅
- [x] Synthetic data generator matching OpsNexus MetricPayload
- [x] Basic data pipeline (ingest → clean → feature engineer)
- [x] Simple baseline model (linear regression for CPU prediction)

### Phase 2: Core ML 🔄
- [x] Multi-metric prediction framework (CPU + memory ready)
- [ ] Anomaly detection (Isolation Forest/Autoencoder - planned)
- [x] Inference API endpoints (/predict, /detect-anomaly)
- [ ] Basic visualization (actual vs predicted plots - planned)

### Phase 3: Integration Readiness 🔄
- [ ] API client for OpsNexus `/analytics` endpoint (configurable)
- [x] Model versioning and persistence
- [x] Comprehensive evaluation metrics (MAE, RMSE)
- [x] Deployment steps documented
- [ ] OpsNexus integration guide (planned)

## 📝 Implementation Notes

### Design Decisions
1. **Model Choice**: Started with linear regression for interpretability and simplicity
2. **Feature Engineering**: Time-based, lagged, rolling statistics, and interaction features
3. **API Design**: RESTful JSON API following common microservice patterns
4. **Error Handling**: Graceful degradation with informative error messages
5. **Testing**: Comprehensive unit tests for each component

### Limitations & Future Work
1. **Single Metric Focus**: Currently focused on CPU usage prediction
2. **Simple Anomaly Detection**: Uses prediction residuals (to be enhanced)
3. **Batch Learning**: Online learning capabilities planned for future
4. **Model Registry**: Will implement proper model versioning and A/B testing

## 🙋‍♂️ Support

For questions, issues, or contributions:
1. Check the existing documentation
2. Review the source code comments
3. Create an issue in the repository

---

**OpsNexus-ML**: Transforming system monitoring from observation to intelligent anticipation.