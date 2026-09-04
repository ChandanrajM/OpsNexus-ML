# OpsNexus-ML Manual Verification Guide

This guide will help you manually verify that all components of the OpsNexus-ML system are working correctly after our enhancements.

## 📋 Table of Contents
1. [System Prerequisites](#1-system-prerequisites)
2. [API Service Verification](#2-api-service-verification)
3. [Prediction Endpoint Verification](#3-prediction-endpoint-verification)
4. [Anomaly Detection Verification](#4-anomaly-detection-verification)
5. [Model Information Verification](#5-model-information-verification)
6. [Model Comparison Verification](#6-model-comparison-verification)
7. [Enhanced Prediction Verification](#7-enhanced-prediction-verification)
8. [Dashboard Verification](#8-dashboard-verification)
9. [GitHub Actions Workflow Verification](#9-github-actions-workflow-verification)
10. [End-to-End Integration Test](#10-end-to-end-integration-test)

---

## 1. System Prerequisites

Before starting verification, ensure:

```bash
# Check that you're in the correct directory
pwd
# Should show: /home/chandanraj-m/OpsNexus-ML

# Check that the virtual environment is activated
which python
# Should show something like: /home/chandanraj-m/OpsNexus-ML/venv/bin/python

# Check that telemetry data exists from opsnexus-agent
ls -la /home/chandanraj-m/opsnexus-local-data/telemetry.json
# Should show the file exists and is recent
```

## 2. API Service Verification

### 2.1 Check if API is Running
```bash
# Check if the API service is running on port 5000
curl -s http://localhost:5000/health

# Expected response:
# {
#   "model_loaded":true,
#   "service":"OpsNexus-ML API",
#   "status":"healthy",
#   "timestamp":"2026-09-04T20:28:29.708072Z"
# }
```

### 2.2 Start API if Not Running
If the health check fails, start the API:
```bash
source venv/bin/activate
python api/app.py &
# Wait a few seconds for it to start
```

## 3. Prediction Endpoint Verification

### 3.1 Test Basic CPU Prediction
```bash
curl -s -X POST http://localhost:5000/predict/cpu \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"local-agent"}'

# Expected response structure:
# {
#   "agent_id":"local-agent",
#   "confidence_interval":{"lower":0,"upper":10},
#   "horizon_minutes":10,
#   "model_info":{
#     "features_used":61,
#     "model_loaded":true,
#     "model_type":"linear_regression",
#     "training_metrics":{...}
#   },
#   "predicted_cpu_usage_percent":3.82,
#   "prediction_timestamp":"2026-09-04T20:43:02.207325Z"
# }
```

### 3.2 Verify Prediction Values are Realistic
- The `predicted_cpu_usage_percent` should be between 0-100
- For an idle system, values typically range from 1-10%
- The confidence interval should be reasonable (typically ±5-10 points)

### 3.3 Test with Custom Parameters
```bash
curl -s -X POST http://localhost:5000/predict/cpu \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id":"local-agent",
    "horizon_minutes":5,
    "lookback_points":50
  }'
```

## 4. Anomaly Detection Verification

### 4.1 Test Anomaly Detection Endpoint
```bash
curl -s -X POST http://localhost:5000/detect/anomaly \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id":"local-agent",
    "lookback_minutes":30,
    "sensitivity":"medium"
  }'

# Expected response structure:
# {
#   "agent_id":"local-agent",
#   "detection_timestamp":"2026-09-04T20:45:12.123456Z",
#   "lookback_minutes":30,
#   "sensitivity":"medium",
#   "anomaly_score":0.15,
#   "is_anomaly":false,
#   "confidence":0.85,
#   "contributing_factors":[
#     {"feature":"cpu_usage_percent","deviation_score":0.2,"value":2.5,"importance":0.3},
#     ...
#   ],
#   "model_info":{
#     "model_type":"isolation_forest",
#     "detection_method":"isolation_forest",
#     "model_loaded":true,
#     "n_estimators":200,
#     "features_used":61,
#     "explainability_available":true
#   }
# }
```

### 4.2 Verify Anomaly Score Logic
- `anomaly_score` should be between 0-1
- `is_anomaly` should be `true` when score > 0.7 (threshold)
- `confidence` should be between 0-1
- `contributing_factors` should be an array of objects with feature names and scores

## 5. Model Information Verification

### 5.1 Get Basic Model Info
```bash
curl -s http://localhost:5000/models/info
```

### 5.2 Get Enhanced Model Info
```bash
curl -s http://localhost:5000/models/info/enhanced

# Expected response should show:
# {
#   "model_loaded":true,
#   "timestamp":"2026-09-04T20:46:01.234567Z",
#   "model_type":"linear_regression",
#   "feature_count":61,
#   "feature_names":[list of 61 feature names],
#   "training_metrics":{
#     "model_type":"linear_regression",
#     "train_mae":0.47257,
#     "test_mae":0.52506,
#     "train_r2":0.30675,
#     "test_r2":0.23686,
#     ...
#   }
# }
```

### 5.3 Verify Model Details
- `model_loaded` should be `true`
- `feature_count` should be `61` (for real-data model)
- `training_metrics` should contain actual values (not dummy values like 2.1, 3.5, 0.85)
- `feature_names` should contain the expected 61 features we engineered

## 6. Model Comparison Verification

### 6.1 Test Model Comparison Endpoint
```bash
curl -s -X POST http://localhost:5000/models/compare \
  -H "Content-Type: application/json" \
  -d '{"lookback_points":100}'

# Expected response should show comparison of all 4 models:
# {
#   "comparison_timestamp":"2026-09-04T20:47:12.123456Z",
#   "lookback_points":100,
#   "training_samples":420,
#   "test_samples":105,
#   "feature_count":61,
#   "best_model":{
#     "model_type":"random_forest",
#     "test_mae":0.42
#   },
#   "all_models":{
#     "linear_regression":{"status":"success","test_mae":0.53,"test_rmse":0.85,"test_r2":0.24},
#     "ridge":{"status":"success","test_mae":0.51,"test_rmse":0.82,"test_r2":0.26},
#     "lasso":{"status":"success","test_mae":0.55,"test_rmse":0.88,"test_r2":0.20},
#     "random_forest":{"status":"success","test_mae":0.42,"test_rmse":0.68,"test_r2":0.45}
#   },
#   "recommendation":"Use random_forest for best performance (MAE: 0.4200)"
# }
```

### 6.2 Verify Comparison Logic
- All 4 models should be tested: linear_regression, ridge, lasso, random_forest
- Each should have a status of "success" (unless there was an error)
- The best_model should be the one with lowest test_mae
- The recommendation should match the best_model type

## 7. Enhanced Prediction Verification

### 7.1 Test Enhanced Prediction Endpoint
```bash
curl -s -X POST http://localhost:5000/predict/enhanced \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id":"local-agent",
    "horizon_minutes":10,
    "lookback_points":100,
    "model_type":"auto"
  }'

# Expected response should include:
# {
#   "agent_id":"local-agent",
#   "prediction_timestamp":"2026-09-04T20:48:12.123456Z",
#   "horizon_minutes":10,
#   "predicted_cpu_usage_percent":3.91,
#   "confidence_interval":{"lower":0.5,"upper":7.3},
#   "model_info":{
#     "features_used":61,
#     "model_loaded":true,
#     "requested_model_type":"auto",
#     "actual_model_type":"linear_regression",
#     "training_metrics":{...}
#   }
# }
```

### 7.2 Test Specific Model Types
Try different model types:
```bash
# Test linear_regression
curl -s -X POST http://localhost:5000/predict/enhanced \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"local-agent","model_type":"linear_regression"}'

# Test ridge
curl -s -X POST http://localhost:5000/predict/enhanced \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"local-agent","model_type":"ridge"}'
```

## 8. Dashboard Verification

### 8.1 Start Text-Based Dashboard
```bash
# In one terminal window
cd dashboard
python realtime_dashboard.py
```

You should see a continuously updating display showing:
- System health status
- Current metrics (CPU, memory, etc.)
- ML predictions
- Anomaly detection status
- Performance metrics
- Trends

### 8.2 Start Web-Based Dashboard
```bash
# In another terminal window
cd dashboard
python launcher.py
# Choose option 2 for web-based dashboard
# Or directly:
python -m http.server 8080
```

Then open in browser: http://localhost:8080

You should see:
- Interactive charts showing CPU usage trends
- Status cards for health, metrics, prediction, and anomaly detection
- Real-time updating values
- Responsive design that works on mobile/desktop

### 8.3 Test Dashboard Data Flow
While the dashboards are running:
1. Check that values update every 5 seconds
2. Verify that CPU usage numbers match what you see in the API
3. Check that prediction values change over time
4. Verify that anomaly detection shows reasonable scores

## 9. GitHub Actions Workflow Verification

### 9.1 Check Workflow File
```bash
# View the enhanced workflow
cat .github/workflows/cd.yml

# Look for these key enhancements:
# - "Fetch training data from local telemetry" step
# - Real data processing in training steps
# - Proper model registration with actual metrics
# - Fallback to synthetic data
```

### 9.2 Trigger Workflow Manually (Optional)
You can trigger the workflow manually via GitHub UI:
1. Go to your repository on GitHub
2. Click "Actions" tab
3. Select "Continuous Deployment" workflow
4. Click "Run workflow"
5. Choose environment (staging/production) or let it run model-training-pipeline
6. Monitor the logs to see:
   - Telemetry data being fetched
   - Models being trained on real data
   - Actual metrics being registered
   - Docker images being built and pushed

## 10. End-to-End Integration Test

### 10.1 Complete Data Flow Test
Let's trace data from opsnexus-agent through to prediction:

```bash
# 1. Check that telemetry data is being generated
ls -la /home/chandanraj-m/opsnexus-local-data/telemetry.json
# Should show recent modification time

# 2. Look at a sample of the data
head -5 /home/chandanraj-m/opsnexus-local-data/telemetry.json | jq '.'
# Should show JSON with agent_id, timestamp, metrics.system, etc.

# 3. Verify API can process this data
curl -s -X POST http://localhost:5000/predict/cpu \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"local-agent"}'
# Should return a reasonable prediction

# 4. Check that the prediction makes sense given recent data
# Get recent actual CPU usage
recent_cpu=$(tail -1 /home/chandanraj-m/opsnexus-local-data/telemetry.json | 
  jq '.metrics.system.cpu.usage_percent')
echo "Recent actual CPU: $recent_cpu%"

# Compare with prediction (should be reasonably close)
```

### 10.2 Test Model Retraining Readiness
```bash
# Check that the real-data model exists and is being used
ls -la models/cpu_predictor_real_data.pkl
# Should exist and be reasonably sized

# Check that the API is using it
curl -s http://localhost:5000/models/info/enhanced | jq '.model_type, .feature_count'
# Should show "linear_regression" and 61
```

### 10.3 Test Anomaly Detection with Real Data
```bash
# Test anomaly detection with real telemetry data
curl -s -X POST http://localhost:5000/detect/anomaly \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"local-agent","lookback_minutes":10}'
# Should return a reasonable anomaly score (typically low for normal operation)
```

## ✅ Verification Complete

If all these checks pass, your OpsNexus-ML system is working correctly with:

1. ✅ **Fixed prediction endpoint** - uses most recent telemetry sample
2. ✅ **Enhanced model loading** - prioritizes real-data 61-feature model
3. ✅ **Working anomaly detection** - with explainability and contributing factors
4. ✅ **Comprehensive model comparison** - tests all 4 algorithms
5. ✅ **Enhanced prediction endpoint** - supports specific model types
6. ✅ **Real-time dashboards** - both terminal and web-based
7. ✅ **Automated retraining pipeline** - uses real telemetry data in GitHub Actions
8. ✅ **End-to-end integration** - opsnexus-agent → telemetry → pipeline → models → API

## 🔧 Troubleshooting

If any checks fail:

1. **API not responding**: 
   ```bash
   # Check if port 5000 is in use
   lsof -i :5000
   # Kill existing process if needed
   kill -9 $(lsof -t -i:5000)
   # Restart API
   source venv/bin/activate && python api/app.py &
   ```

2. **No telemetry data**:
   ```bash
   # Check if opsnexus-agent is running
   ps aux | grep opsnexus-agent
   # Start it if needed
   ```

3. **Model not loading**:
   ```bash
   # Check model files exist
   ls -la models/
   # Check permissions
   # Try loading manually:
   source venv/bin/activate
   python -c "from models.enhanced_model import EnhancedCPUUsagePredictor; m=EnhancedCPUUsagePredictor(); m.load_model('models/cpu_predictor_real_data.pkl'); print('Model loaded:', m.is_trained)"
   ```

4. **Dashboard issues**:
   ```bash
   # Check required packages
   pip install matplotlib requests
   ```

For detailed logs, check:
- API logs: `/tmp/api.log` or stdout where you started it
- Dashboard output: terminal where it's running
- GitHub Actions: Actions tab in your GitHub repository