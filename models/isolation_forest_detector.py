"""
Isolation Forest Anomaly Detector for OpsNexus-ML
Implements advanced anomaly detection using sklearn.ensemble.IsolationForest
"""
import numpy as np
import pickle
import os
from typing import Tuple, Optional, List, Dict
import logging
from sklearn.ensemble import IsolationForest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IsolationForestDetector:
    """
    Isolation Forest model for detecting anomalies in telemetry feature space
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Isolation Forest detector

        Args:
            model_path: Path to load/save the model (optional)
        """
        self.model_path = model_path
        self.estimator = None
        self.feature_names = None
        self.is_trained = False
        self._score_bounds = None  # To normalize scores to [0,1]

    def train(self, X: np.ndarray, y=None, feature_names: Optional[List[str]] = None,
              contamination: float = 0.1, n_estimators: int = 100,
              max_samples: str = 'auto', random_state: int = 42) -> dict:
        """
        Train the Isolation Forest model

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Ignored (present for API consistency with supervised models)
            feature_names: Names of features (for interpretability)
            contamination: Expected proportion of outliers in the data
            n_estimators: Number of base estimators in the ensemble
            max_samples: Number of samples to draw for each base estimator
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training Isolation Forest detector with {X.shape[0]} samples and {X.shape[1]} features")

        # Store feature names
        self.feature_names = feature_names

        # Initialize and train the Isolation Forest
        self.estimator = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1
        )
        self.estimator.fit(X)
        self.is_trained = True

        # Calculate score bounds for normalization to [0,1] (Isolation Forest returns negative scores)
        scores = self.estimator.decision_function(X)
        self._score_bounds = (np.min(scores), np.max(scores))
        score_range = self._score_bounds[1] - self._score_bounds[0]
        if score_range == 0:
            score_range = 1.0  # Avoid division by zero

        metrics = {
            'n_estimators': n_estimators,
            'contamination': contamination,
            'max_samples': max_samples,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'score_min_raw': float(np.min(scores)),
            'score_max_raw': float(np.max(scores)),
            'score_mean_raw': float(np.mean(scores))
        }

        logger.info(f"Training completed. Score range: [{self._score_bounds[0]:.4f}, {self._score_bounds[1]:.4f}]")

        return metrics

    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores for samples (higher = more anomalous)

        Args:
            X: Feature matrix

        Returns:
            Array of anomaly scores in [0,1] range (1 = most anomalous)
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")

        # Isolation Forest returns negative scores; normalize to [0,1]
        raw_scores = self.estimator.decision_function(X)
        min_score, max_score = self._score_bounds
        score_range = max_score - min_score
        if score_range == 0:
            score_range = 1.0

        # Convert: (score - min) / (max - min) -> [0,1], then invert so HIGHER score = MORE anomalous
        # Isolation Forest: MORE NEGATIVE = more anomalous
        normalized = (raw_scores - min_score) / score_range
        anomaly_scores = 1.0 - normalized  # Flip so 1 = most anomalous

        return anomaly_scores

    def detect_anomaly(self, X: np.ndarray, threshold: float = 0.7) -> np.ndarray:
        """
        Detect anomalies based on score threshold

        Args:
            X: Feature matrix
            threshold: Anomaly threshold (0-1, higher = fewer anomalies)

        Returns:
            Boolean array where True indicates anomaly
        """
        scores = self.predict_anomaly_score(X)
        return scores >= threshold

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance (average decrease in impurity across all trees in the forest)

        Returns:
            Dictionary mapping feature names to importance scores, or None if not trained
        """
        if not self.is_trained or self.estimator is None or not hasattr(self.estimator, 'estimators_'):
            return None

        if self.feature_names is None:
            return None

        # Collect feature importances from all trees
        importances = []
        for estimator in self.estimator.estimators_:
            # Each estimator is a DecisionTreeRegressor, which has feature_importances_
            importances.append(estimator.feature_importances_)

        # Average across all trees
        avg_importances = np.mean(importances, axis=0)

        # Map to feature names
        importance_dict = {}
        for name, imp in zip(self.feature_names, avg_importances):
            importance_dict[name] = float(imp)

        return importance_dict

    def explain_anomaly(self, X: np.ndarray, sample_idx: int = None, top_n: int = 5) -> Optional[Dict]:
        """
        Explain anomaly detection results for a specific sample or all samples

        Args:
            X: Feature matrix (n_samples, n_features)
            sample_idx: Index of sample to explain (if None, returns explanation for all samples)
            top_n: Number of top contributing features to return

        Returns:
            Dictionary with explanation details or None if not trained
        """
        if not self.is_trained:
            return None

        if self.feature_names is None:
            return None

        # Get anomaly scores and decision function values
        raw_scores = self.estimator.decision_function(X)
        anomaly_scores = self.predict_anomaly_score(X)

        if sample_idx is not None:
            # Explain specific sample
            if sample_idx >= len(X) or sample_idx < 0:
                return None

            # For Isolation Forest, we can approximate feature contributions
            # by looking at how much each feature deviates from the norm
            # This is a simplified explanation - in practice, you might use SHAP values
            sample = X[sample_idx]

            # Calculate z-score like deviation for each feature (simplified approach)
            feature_means = np.mean(X, axis=0)
            feature_stds = np.std(X, axis=0)
            feature_stds = np.where(feature_stds == 0, 1, feature_stds)  # Avoid division by zero

            deviations = np.abs((sample - feature_means) / feature_stds)

            # Get top contributing features
            top_indices = np.argsort(deviations)[::-1][:top_n]

            contributing_factors = []
            for idx in top_indices:
                contributing_factors.append({
                    'feature': self.feature_names[idx],
                    'deviation_score': float(deviations[idx]),
                    'value': float(sample[idx]),
                    'importance': float(self.get_feature_importance().get(self.feature_names[idx], 0.0)) if self.get_feature_importance() else 0.0
                })

            return {
                'sample_index': int(sample_idx),
                'anomaly_score': float(anomaly_scores[sample_idx]),
                'is_anomaly': bool(anomaly_scores[sample_idx] >= 0.7),  # Using default threshold
                'raw_score': float(raw_scores[sample_idx]),
                'top_contributing_factors': contributing_factors
            }
        else:
            # Explain all samples (return summary)
            return {
                'n_samples': int(len(X)),
                'anomaly_score_mean': float(np.mean(anomaly_scores)),
                'anomaly_score_std': float(np.std(anomaly_scores)),
                'n_anomalies': int(np.sum(anomaly_scores >= 0.7)),
                'anomaly_percentage': float(np.sum(anomaly_scores >= 0.7) / len(X) * 100),
                'feature_importance': self.get_feature_importance()
            }

    def save_model(self, filepath: Optional[str] = None) -> None:
        """
        Save the trained model to disk

        Args:
            filepath: Path to save the model (uses instance path if not provided)
        """
        if filepath is None:
            filepath = self.model_path

        if filepath is None:
            raise ValueError("No filepath provided for saving model")

        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'estimator': self.estimator,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'score_bounds': self._score_bounds
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: Optional[str] = None) -> None:
        """
        Load a trained model from disk

        Args:
            filepath: Path to load the model from (uses instance path if not provided)
        """
        if filepath is None:
            filepath = self.model_path

        if filepath is None:
            raise ValueError("No filepath provided for loading model")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.estimator = model_data['estimator']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        self._score_bounds = model_data['score_bounds']

        logger.info(f"Model loaded from {filepath}")

def create_and_train_detector(X: np.ndarray, y=None,
                             feature_names: Optional[List[str]] = None,
                             model_path: Optional[str] = None,
                             **kwargs) -> IsolationForestDetector:
    """
    Convenience function to create, train, and return an Isolation Forest detector

    Args:
        X: Feature matrix
        y: Ignored (for API consistency)
        feature_names: Names of features
        model_path: Path to save the trained model (optional)
        **kwargs: Additional arguments to pass to train()

    Returns:
        Trained IsolationForestDetector instance
    """
    detector = IsolationForestDetector(model_path=model_path)
    metrics = detector.train(X, y, feature_names=feature_names, **kwargs)

    if model_path is not None:
        detector.save_model(model_path)

    return detector, metrics

def main():
    """
    Example usage of the Isolation Forest detector
    """
    # This would typically be called after preparing data with the pipeline
    print("Isolation Forest Anomaly Detector - Example Usage")
    print("=" * 50)

    # Example with dummy data (in practice, this comes from the data pipeline)
    # Simulate 1000 samples with 10 features
    np.random.seed(42)
    X_example = np.random.randn(1000, 10)
    # Inject some obvious anomalies
    X_example[0:5, :] += 5.0  # Make first 5 samples extreme outliers

    feature_names_example = [
        'cpu_lag_1', 'cpu_lag_2', 'memory_lag_1', 'memory_lag_2',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'cpu_rolling_mean_3', 'memory_rolling_mean_3'
    ]

    # Create and train the detector
    detector, metrics = create_and_train_detector(
        X_example, y=None,
        feature_names=feature_names_example,
        model_path="/home/chandanraj-m/OpsNexus-ML/models/isolation_forest_detector.pkl",
        contamination=0.05  # Expect 5% anomalies
    )

    # Show results
    print("\nTraining Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Show feature importance
    importance = detector.get_feature_importance()
    print("\nFeature Importance (average decrease in impurity):")
    if importance:
        for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {imp:.4f}")
    else:
        print("  Feature importance not available")

    # Detect anomalies
    scores = detector.predict_anomaly_score(X_example)
    anomalies = detector.detect_anomaly(X_example, threshold=0.7)
    n_anomalies = np.sum(anomalies)
    print(f"\nAnomaly Detection:")
    print(f"  Total samples: {len(X_example)}")
    print(f"  Detected anomalies: {n_anomalies} ({n_anomalies/len(X_example)*100:.1f}%)")
    print(f"  Anomaly score range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")

    # Test persistence
    print("\nTesting model persistence...")
    detector.save_model("/home/chandanraj-m/OpsNexus-ML/models/isolation_forest_detector_test.pkl")
    detector_loaded = IsolationForestDetector(model_path="/home/chandanraj-m/OpsNexus-ML/models/isolation_forest_detector_test.pkl")
    detector_loaded.load_model("/home/chandanraj-m/OpsNexus-ML/models/isolation_forest_detector_test.pkl")
    print("✅ Model persistence test passed")

if __name__ == "__main__":
    main()