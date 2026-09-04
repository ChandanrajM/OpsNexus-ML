"""
Enhanced machine learning models for OpsNexus-ML
Implements multiple algorithms for CPU usage prediction with automatic model selection
"""
import numpy as np
import pickle
import os
from typing import Tuple, Optional, Dict, Any, List
import logging
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType:
    """Enumeration of available model types"""
    LINEAR_REGRESSION = "linear_regression"
    RIDGE = "ridge"
    LASSO = "lasso"
    RANDOM_FOREST = "random_forest"


class EnhancedCPUUsagePredictor:
    """
    Enhanced CPU usage predictor that supports multiple algorithms
    and automatic model selection based on performance
    """

    def __init__(self, model_type: str = ModelType.LINEAR_REGRESSION,
                 model_path: Optional[str] = None):
        """
        Initialize the enhanced CPU usage predictor

        Args:
            model_type: Type of model to use (linear_regression, ridge, lasso, random_forest)
            model_path: Path to load/save the model (optional)
        """
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.training_metrics = {}

        # Initialize the model based on type
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the scikit-learn model based on model_type"""
        if self.model_type == ModelType.LINEAR_REGRESSION:
            self.model = LinearRegression()
        elif self.model_type == ModelType.RIDGE:
            self.model = Ridge(alpha=1.0, random_state=42)
        elif self.model_type == ModelType.LASSO:
            self.model = Lasso(alpha=0.1, random_state=42, max_iter=1000)
        elif self.model_type == ModelType.RANDOM_FOREST:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, X: np.ndarray, y: np.ndarray,
              feature_names: Optional[List[str]] = None,
              test_size: float = 0.2,
              random_state: int = 42,
              use_cross_validation: bool = True) -> Dict[str, Any]:
        """
        Train the selected model

        Args:
            X: Feature matrix
            y: Target values (CPU usage percentages)
            feature_names: Names of features (for interpretability)
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            use_cross_validation: Whether to use cross-validation for better metrics

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} model with {X.shape[0]} samples and {X.shape[1]} features")

        # Store feature names
        self.feature_names = feature_names

        # Set random seed for reproducibility
        np.random.seed(random_state)

        # Split data
        n_samples = X.shape[0]
        n_test = int(n_samples * test_size)

        # Create random indices for shuffling
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        # Cross-validation score (if requested)
        cv_scores = None
        if use_cross_validation and len(X_train) >= 10:
            try:
                cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
                cv_mae = -cv_scores.mean()
                cv_std = cv_scores.std()
            except Exception as e:
                logger.warning(f"Cross-validation failed: {e}")
                cv_mae = None
                cv_std = None
        else:
            cv_mae = None
            cv_std = None

        metrics = {
            'model_type': self.model_type,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'cv_mae': cv_mae,
            'cv_mae_std': cv_std,
            'feature_count': X.shape[1]
        }

        self.training_metrics = metrics

        logger.info(f"Training completed. Test MAE: {test_mae:.4f}, Test R²: {test_r2:.4f}")

        return metrics

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

        return self.model.predict(X)

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

        prediction = self.model.predict(latest_features)
        return float(prediction[0])

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance (varies by model type)

        Returns:
            Dictionary mapping feature names to importance scores, or None if not trained
        """
        if not self.is_trained or self.feature_names is None:
            return None

        importance = {}

        try:
            if self.model_type in [ModelType.LINEAR_REGRESSION, ModelType.RIDGE, ModelType.LASSO]:
                # For linear models, coefficients represent feature importance
                if hasattr(self.model, 'coef_'):
                    for name, coef in zip(self.feature_names, self.model.coef_):
                        importance[name] = float(coef)
                # Add intercept
                if hasattr(self.model, 'intercept_'):
                    importance['intercept'] = float(self.model.intercept_)

            elif self.model_type == ModelType.RANDOM_FOREST:
                # For tree-based models, use feature_importances_
                if hasattr(self.model, 'feature_importances_'):
                    for name, importance_val in zip(self.feature_names, self.model.feature_importances_):
                        importance[name] = float(importance_val)

        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return None

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
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics
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

        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        self.training_metrics = model_data.get('training_metrics', {})

        logger.info(f"Model loaded from {filepath}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model

        Returns:
            Dictionary with model information
        """
        if not self.is_trained:
            return {'error': 'Model not trained', 'is_trained': False}

        info = {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics
        }

        # Add model-specific information
        if hasattr(self.model, 'get_params'):
            info['model_parameters'] = self.model.get_params()

        return info


def create_and_train_model(X: np.ndarray, y: np.ndarray,
                          model_type: str = ModelType.LINEAR_REGRESSION,
                          feature_names: Optional[List[str]] = None,
                          model_path: Optional[str] = None,
                          **kwargs) -> Tuple[EnhancedCPUUsagePredictor, Dict[str, Any]]:
    """
    Convenience function to create, train, and return an enhanced CPU usage predictor

    Args:
        X: Feature matrix
        y: Target values
        model_type: Type of model to use
        feature_names: Names of features
        model_path: Path to save the trained model (optional)
        **kwargs: Additional arguments to pass to train()

    Returns:
        Tuple of (trained EnhancedCPUUsagePredictor instance, training metrics)
    """
    predictor = EnhancedCPUUsagePredictor(model_type=model_type, model_path=model_path)
    metrics = predictor.train(X, y, feature_names=feature_names, **kwargs)

    if model_path is not None:
        predictor.save_model(model_path)

    return predictor, metrics


def compare_models(X: np.ndarray, y: np.ndarray,
                  feature_names: Optional[List[str]] = None,
                  test_size: float = 0.2,
                  random_state: int = 42) -> Dict[str, Any]:
    """
    Compare all available models and return the best performing one

    Args:
        X: Feature matrix
        y: Target values
        feature_names: Names of features
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with comparison results and the best model
    """
    logger.info("Comparing all available models...")

    models_to_compare = [
        ModelType.LINEAR_REGRESSION,
        ModelType.RIDGE,
        ModelType.LASSO,
        ModelType.RANDOM_FOREST
    ]

    results = {}
    best_model = None
    best_score = float('inf')  # We want to minimize MAE
    best_model_type = None

    for model_type in models_to_compare:
        logger.info(f"Evaluating {model_type}...")

        try:
            predictor = EnhancedCPUUsagePredictor(model_type=model_type)
            metrics = predictor.train(
                X, y,
                feature_names=feature_names,
                test_size=test_size,
                random_state=random_state,
                use_cross_validation=True
            )

            # Use test MAE as the primary metric for comparison
            test_mae = metrics.get('test_mae', float('inf'))
            results[model_type] = {
                'predictor': predictor,
                'metrics': metrics,
                'test_mae': test_mae
            }

            if test_mae < best_score:
                best_score = test_mae
                best_model = predictor
                best_model_type = model_type

            logger.info(f"{model_type} - Test MAE: {test_mae:.4f}")

        except Exception as e:
            logger.error(f"Failed to evaluate {model_type}: {e}")
            results[model_type] = {
                'error': str(e),
                'test_mae': float('inf')
            }

    logger.info(f"Best model: {best_model_type} with Test MAE: {best_score:.4f}")

    comparison_result = {
        'best_model': best_model,
        'best_model_type': best_model_type,
        'best_test_mae': best_score,
        'all_results': results
    }

    return comparison_result


def main():
    """
    Example usage of the enhanced models
    """
    print("Enhanced CPU Usage Predictor - Example Usage")
    print("=" * 50)

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

    # Compare all models
    print("\nModel Comparison:")
    print("-" * 30)
    comparison = compare_models(
        X_example, y_example,
        feature_names=feature_names_example
    )

    # Show results for each model
    for model_type, result in comparison['all_results'].items():
        if 'error' in result:
            print(f"{model_type:20} - ERROR: {result['error']}")
        else:
            metrics = result['metrics']
            print(f"{model_type:20} - Test MAE: {metrics['test_mae']:.4f}, Test R²: {metrics['test_r2']:.4f}")

    print(f"\nBest Model: {comparison['best_model_type']}")
    print(f"Best Test MAE: {comparison['best_test_mae']:.4f}")

    # Show feature importance for the best model
    if comparison['best_model']:
        importance = comparison['best_model'].get_feature_importance()
        if importance:
            print("\nFeature Importance (Top 10):")
            sorted_importance = sorted(
                [(k, v) for k, v in importance.items() if k != 'intercept'],
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10]
            for feature, coef in sorted_importance:
                print(f"  {feature:25} : {coef:.4f}")

    # Make a prediction with the best model
    sample_features = X_example[0:1]  # First sample
    if comparison['best_model']:
        prediction = comparison['best_model'].predict_next_cpu_usage(sample_features)
        actual = y_example[0]
        print(f"\nSample Prediction (using best model):")
        print(f"  Predicted CPU usage: {prediction:.2f}%")
        print(f"  Actual CPU usage: {actual:.2f}%")
        print(f"  Error: {abs(prediction - actual):.2f}%")


if __name__ == "__main__":
    main()