"""
Baseline machine learning model for OpsNexus-ML
Implements simple linear regression for CPU usage prediction using numpy only
"""
import numpy as np
import pickle
import os
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CPUUsagePredictor:
    """
    Simple linear regression model for predicting CPU usage using numpy
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the CPU usage predictor

        Args:
            model_path: Path to load/save the model (optional)
        """
        self.model_path = model_path
        self.coefficients = None
        self.intercept = None
        self.is_trained = False
        self.feature_names = None

    def train(self, X: np.ndarray, y: np.ndarray,
              feature_names: Optional[list] = None,
              test_size: float = 0.2,
              random_state: int = 42) -> dict:
        """
        Train the linear regression model using normal equation

        Args:
            X: Feature matrix
            y: Target values (CPU usage percentages)
            feature_names: Names of features (for interpretability)
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training CPU usage predictor with {X.shape[0]} samples and {X.shape[1]} features")

        # Store feature names
        self.feature_names = feature_names

        # Set random seed for reproducibility
        np.random.seed(random_state)

        # Split data manually
        n_samples = X.shape[0]
        n_test = int(n_samples * test_size)

        # Create random indices for shuffling
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Train the model using normal equation: θ = (X^T X)^(-1) X^T y
        # Add bias term to X_train
        X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

        # Calculate coefficients using normal equation
        try:
            theta = np.linalg.inv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse
            theta = np.linalg.pinv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train

        self.intercept = theta[0]
        self.coefficients = theta[1:]
        self.is_trained = True

        # Make predictions
        y_pred_train = self._predict_with_bias(X_train)
        y_pred_test = self._predict_with_bias(X_test)

        # Calculate metrics
        train_mae = np.mean(np.abs(y_train - y_pred_train))
        test_mae = np.mean(np.abs(y_test - y_pred_test))
        train_rmse = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
        test_rmse = np.sqrt(np.mean((y_test - y_pred_test) ** 2))
        train_r2 = 1 - (np.sum((y_train - y_pred_train) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2))
        test_r2 = 1 - (np.sum((y_test - y_pred_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

        metrics = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }

        logger.info(f"Training completed. Test MAE: {test_mae:.4f}, Test R²: {test_r2:.4f}")

        return metrics

    def _predict_with_bias(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions adding bias term

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_bias @ np.hstack([self.intercept, self.coefficients])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")

        return self._predict_with_bias(X)

    def predict_next_cpu_usage(self, latest_features: np.ndarray) -> float:
        """
        Predict the next CPU usage value from the latest feature vector

        Args:
            latest_features: Feature vector for the most recent observation

        Returns:
            Predicted CPU usage percentage
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train() first.")

        # Ensure the input is 2D (single sample)
        if len(latest_features.shape) == 1:
            latest_features = latest_features.reshape(1, -1)

        prediction = self._predict_with_bias(latest_features)
        return float(prediction[0])

    def get_feature_importance(self) -> Optional[dict]:
        """
        Get feature importance (coefficients for linear regression)

        Returns:
            Dictionary mapping feature names to coefficients, or None if not trained
        """
        if not self.is_trained or self.feature_names is None:
            return None

        # For linear regression, coefficients represent feature importance
        importance = {}
        for name, coef in zip(self.feature_names, self.coefficients):
            importance[name] = float(coef)

        # Add intercept
        importance['intercept'] = float(self.intercept)

        return importance

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
            'coefficients': self.coefficients,
            'intercept': self.intercept,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
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

        self.coefficients = model_data['coefficients']
        self.intercept = model_data['intercept']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']

        logger.info(f"Model loaded from {filepath}")

def create_and_train_model(X: np.ndarray, y: np.ndarray,
                          feature_names: Optional[list] = None,
                          model_path: Optional[str] = None) -> CPUUsagePredictor:
    """
    Convenience function to create, train, and return a CPU usage predictor

    Args:
        X: Feature matrix
        y: Target values
        feature_names: Names of features
        model_path: Path to save the trained model (optional)

    Returns:
        Trained CPUUsagePredictor instance
    """
    predictor = CPUUsagePredictor(model_path=model_path)
    metrics = predictor.train(X, y, feature_names=feature_names)

    if model_path is not None:
        predictor.save_model(model_path)

    return predictor, metrics

def main():
    """
    Example usage of the baseline model
    """
    # This would typically be called after preparing data with the pipeline
    print("CPU Usage Predictor - Example Usage")
    print("=" * 40)

    # Example with dummy data (in practice, this comes from the data pipeline)
    # Simulate 1000 samples with 10 features
    np.random.seed(42)
    X_example = np.random.randn(1000, 10)
    # Create a target that has some relationship with the features
    y_example = (X_example[:, 0] * 2.5 + X_example[:, 1] * -1.2 +
                 np.random.randn(1000) * 0.5 + 50)  # Center around 50%
    y_example = np.clip(y_example, 0, 100)  # Keep in valid range

    feature_names_example = [
        'cpu_lag_1', 'cpu_lag_2', 'memory_lag_1', 'memory_lag_2',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'cpu_rolling_mean_3', 'memory_rolling_mean_3'
    ]

    # Create and train the model
    predictor, metrics = create_and_train_model(
        X_example, y_example,
        feature_names=feature_names_example,
        model_path="/home/chandanraj-m/OpsNexus-ML/models/cpu_predictor.pkl"
    )

    # Show results
    print("\nTraining Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Show feature importance
    importance = predictor.get_feature_importance()
    print("\nFeature Importance (coefficients):")
    for feature, coef in sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feature}: {coef:.4f}")

    # Make a prediction
    sample_features = X_example[0:1]  # First sample
    prediction = predictor.predict_next_cpu_usage(sample_features)
    actual = y_example[0]
    print(f"\nSample Prediction:")
    print(f"  Predicted CPU usage: {prediction:.2f}%")
    print(f"  Actual CPU usage: {actual:.2f}%")
    print(f"  Error: {abs(prediction - actual):.2f}%")

if __name__ == "__main__":
    main()