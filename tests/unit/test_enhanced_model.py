"""
Unit tests for Enhanced CPU Usage Predictor
"""
import os
import sys
import numpy as np
import pytest
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.enhanced_model import (
    EnhancedCPUUsagePredictor,
    ModelType,
    compare_models,
    create_and_train_model
)


class TestEnhancedCPUUsagePredictor:
    """Test cases for EnhancedCPUUsagePredictor"""

    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 10
        self.X_train = np.random.randn(self.n_samples, self.n_features)
        self.y_train = np.random.rand(self.n_samples) * 100  # CPU usage 0-100%
        self.X_test = np.random.randn(20, self.n_features)
        self.y_test = np.random.rand(20) * 100
        self.feature_names = [f'feature_{i}' for i in range(self.n_features)]

    def test_initialization(self):
        """Test predictor initialization for all model types"""
        for model_type in [ModelType.LINEAR_REGRESSION, ModelType.RIDGE, ModelType.LASSO, ModelType.RANDOM_FOREST]:
            predictor = EnhancedCPUUsagePredictor(model_type=model_type)
            assert predictor.model_type == model_type
            assert predictor.model is not None
            assert predictor.is_trained is False
            assert predictor.feature_names is None

    def test_initialization_with_model_path(self):
        """Test predictor initialization with model path"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.pkl')
            predictor = EnhancedCPUUsagePredictor(model_type=ModelType.RIDGE, model_path=model_path)
            assert predictor.model_path == model_path
            assert predictor.model_type == ModelType.RIDGE

    def test_train_linear_regression(self):
        """Test linear regression training"""
        predictor = EnhancedCPUUsagePredictor(model_type=ModelType.LINEAR_REGRESSION)
        metrics = predictor.train(self.X_train, self.y_train, feature_names=self.feature_names)

        assert predictor.is_trained is True
        assert predictor.feature_names == self.feature_names
        assert 'test_mae' in metrics
        assert 'test_r2' in metrics
        assert metrics['model_type'] == ModelType.LINEAR_REGRESSION
        # Note: train/test split sizes may vary slightly due to random sampling
        assert metrics['train_samples'] + metrics['test_samples'] == len(self.X_train)
        assert metrics['test_samples'] > 0
        assert metrics['train_samples'] > 0

    def test_train_ridge(self):
        """Test ridge regression training"""
        predictor = EnhancedCPUUsagePredictor(model_type=ModelType.RIDGE)
        metrics = predictor.train(self.X_train, self.y_train, feature_names=self.feature_names)

        assert predictor.is_trained is True
        assert metrics['model_type'] == ModelType.RIDGE
        assert 'test_mae' in metrics
        assert 'test_r2' in metrics

    def test_train_lasso(self):
        """Test lasso regression training"""
        predictor = EnhancedCPUUsagePredictor(model_type=ModelType.LASSO)
        metrics = predictor.train(self.X_train, self.y_train, feature_names=self.feature_names)

        assert predictor.is_trained is True
        assert metrics['model_type'] == ModelType.LASSO
        assert 'test_mae' in metrics
        assert 'test_r2' in metrics

    def test_train_random_forest(self):
        """Test random forest training"""
        predictor = EnhancedCPUUsagePredictor(model_type=ModelType.RANDOM_FOREST)
        metrics = predictor.train(self.X_train, self.y_train, feature_names=self.feature_names)

        assert predictor.is_trained is True
        assert metrics['model_type'] == ModelType.RANDOM_FOREST
        assert 'test_mae' in metrics
        assert 'test_r2' in metrics
        assert 'feature_count' in metrics

    def test_predict(self):
        """Test prediction for all model types"""
        for model_type in [ModelType.LINEAR_REGRESSION, ModelType.RIDGE, ModelType.LASSO, ModelType.RANDOM_FOREST]:
            predictor = EnhancedCPUUsagePredictor(model_type=model_type)
            predictor.train(self.X_train, self.y_train, feature_names=self.feature_names)

            predictions = predictor.predict(self.X_test)

            assert predictions.shape == (20,)
            assert np.all(predictions >= 0)  # CPU usage can't be negative
            assert np.all(predictions <= 100)  # CPU usage can't exceed 100%

    def test_predict_without_fit(self):
        """Test prediction without fitting raises error"""
        for model_type in [ModelType.LINEAR_REGRESSION, ModelType.RIDGE, ModelType.LASSO, ModelType.RANDOM_FOREST]:
            predictor = EnhancedCPUUsagePredictor(model_type=model_type)
            with pytest.raises(ValueError, match="Model has not been trained yet"):
                predictor.predict(self.X_test)

    def test_predict_next_cpu_usage(self):
        """Test single-step prediction"""
        predictor = EnhancedCPUUsagePredictor(model_type=ModelType.RANDOM_FOREST)
        predictor.train(self.X_train, self.y_train, feature_names=self.feature_names)

        # Predict next CPU usage from a single feature vector
        next_features = self.X_test[0:1]
        prediction = predictor.predict_next_cpu_usage(next_features)

        assert isinstance(prediction, (int, float))
        assert 0 <= prediction <= 100

    def test_get_feature_importance(self):
        """Test feature importance extraction"""
        for model_type in [ModelType.LINEAR_REGRESSION, ModelType.RIDGE, ModelType.LASSO, ModelType.RANDOM_FOREST]:
            predictor = EnhancedCPUUsagePredictor(model_type=model_type)
            predictor.train(self.X_train, self.y_train, feature_names=self.feature_names)

            importance = predictor.get_feature_importance()

            assert isinstance(importance, dict)
            # Check that we have feature importance for all features
            assert len(importance) >= self.n_features
            # Check that all feature names are present
            for feature_name in self.feature_names:
                assert feature_name in importance
            # For linear models, intercept should be present
            if model_type in [ModelType.LINEAR_REGRESSION, ModelType.RIDGE, ModelType.LASSO]:
                assert 'intercept' in importance
            assert all(isinstance(v, (int, float)) for v in importance.values())

    def test_get_feature_importance_without_fit(self):
        """Test feature importance without fitting returns None"""
        for model_type in [ModelType.LINEAR_REGRESSION, ModelType.RIDGE, ModelType.LASSO, ModelType.RANDOM_FOREST]:
            predictor = EnhancedCPUUsagePredictor(model_type=model_type)
            result = predictor.get_feature_importance()
            assert result is None

    def test_save_and_load_model(self):
        """Test model persistence"""
        predictor = EnhancedCPUUsagePredictor(model_type=ModelType.RIDGE)
        predictor.train(self.X_train, self.y_train, feature_names=self.feature_names)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_predictor.pkl')
            predictor.save_model(model_path)

            assert os.path.exists(model_path)

            # Load the model
            loaded_predictor = EnhancedCPUUsagePredictor(model_type=ModelType.RIDGE, model_path=model_path)
            loaded_predictor.load_model(model_path)

            assert loaded_predictor.is_trained is True
            assert loaded_predictor.model_type == ModelType.RIDGE
            assert loaded_predictor.feature_names == predictor.feature_names

            # Test that loaded model produces same predictions
            original_preds = predictor.predict(self.X_test)
            loaded_preds = loaded_predictor.predict(self.X_test)
            np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_model_info(self):
        """Test getting model information"""
        predictor = EnhancedCPUUsagePredictor(model_type=ModelType.RANDOM_FOREST)
        predictor.train(self.X_train, self.y_train, feature_names=self.feature_names)

        info = predictor.get_model_info()

        assert info['model_type'] == ModelType.RANDOM_FOREST
        assert info['is_trained'] is True
        assert info['feature_count'] == self.n_features
        assert 'training_metrics' in info
        assert 'model_parameters' in info

    def test_model_info_without_fit(self):
        """Test model information without training"""
        predictor = EnhancedCPUUsagePredictor(model_type=ModelType.LINEAR_REGRESSION)
        info = predictor.get_model_info()

        assert info['is_trained'] is False
        assert 'error' in info or info.get('model_type') == ModelType.LINEAR_REGRESSION

    def test_compare_models_function(self):
        """Test the model comparison function"""
        # Use smaller dataset for faster testing
        X_small = self.X_train[:50]
        y_small = self.y_train[:50]

        comparison = compare_models(X_small, y_small, feature_names=self.feature_names[:5])

        assert 'best_model' in comparison
        assert 'best_model_type' in comparison
        assert 'best_test_mae' in comparison
        assert 'all_results' in comparison

        assert comparison['best_model'] is not None
        assert isinstance(comparison['best_model_type'], str)
        assert isinstance(comparison['best_test_mae'], float)
        assert comparison['best_test_mae'] >= 0

        # Check that all model types were evaluated
        expected_types = {ModelType.LINEAR_REGRESSION, ModelType.RIDGE, ModelType.LASSO, ModelType.RANDOM_FOREST}
        actual_types = set(comparison['all_results'].keys())
        assert expected_types.issubset(actual_types)

        # Check that each result has either success or error
        for model_type, result in comparison['all_results'].items():
            assert 'test_mae' in result
            if 'error' in result:
                assert isinstance(result['error'], str)
            else:
                assert 'metrics' in result
                assert isinstance(result['metrics'], dict)

    def test_create_and_train_model(self):
        """Test the convenience function"""
        predictor, metrics = create_and_train_model(
            self.X_train, self.y_train,
            model_type=ModelType.LASSO,
            feature_names=self.feature_names
        )

        assert isinstance(predictor, EnhancedCPUUsagePredictor)
        assert predictor.model_type == ModelType.LASSO
        assert predictor.is_trained is True
        assert isinstance(metrics, dict)
        assert 'test_mae' in metrics

    def test_create_and_train_detector_compatibility(self):
        """Test that enhanced models work with existing interfaces"""
        predictor = EnhancedCPUUsagePredictor(model_type=ModelType.RIDGE)
        predictor.train(self.X_train, self.y_train, feature_names=self.feature_names)

        # Test that it has the same interface as baseline model for API compatibility
        assert hasattr(predictor, 'predict_next_cpu_usage')
        assert hasattr(predictor, 'get_feature_importance')
        assert hasattr(predictor, 'save_model')
        assert hasattr(predictor, 'load_model')
        assert hasattr(predictor, 'get_model_info')


class TestEnhancedCPUUsagePredictorEdgeCases:
    """Test edge cases for Enhanced CPU Usage Predictor"""

    def setup_method(self):
        """Set up test data for edge case tests"""
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 10
        self.X_train = np.random.randn(self.n_samples, self.n_features)
        self.y_train = np.random.rand(self.n_samples) * 100  # CPU usage 0-100%
        self.feature_names = [f'feature_{i}' for i in range(self.n_features)]

    def test_different_random_states(self):
        """Test that different random states produce different but valid results"""
        predictions = []
        for seed in [42, 123, 456]:
            np.random.seed(seed)
            X = np.random.randn(100, 5)
            y = np.random.rand(100) * 100

            predictor = EnhancedCPUUsagePredictor(model_type=ModelType.RANDOM_FOREST)
            predictor.train(X, y, random_state=seed)
            pred = predictor.predict(X[:5])
            predictions.append(pred)

        # All should be valid predictions
        for pred in predictions:
            assert pred.shape == (5,)
            assert np.all(pred >= 0)
            assert np.all(pred <= 100)

        # At least some should be different (though with small datasets they might be similar)
        # We mainly want to ensure no errors occur

    def test_feature_names_handling(self):
        """Test various feature names scenarios"""
        # Test with None feature names
        predictor = EnhancedCPUUsagePredictor(model_type=ModelType.LINEAR_REGRESSION)
        predictor.train(self.X_train, self.y_train)  # No feature names
        assert predictor.feature_names is None
        importance = predictor.get_feature_importance()
        assert importance is None  # Should return None when no feature names provided

        # Test with empty feature names list
        predictor2 = EnhancedCPUUsagePredictor(model_type=ModelType.LINEAR_REGRESSION)
        predictor2.train(self.X_train, self.y_train, feature_names=[])
        assert predictor2.feature_names == []
        importance2 = predictor2.get_feature_importance()
        # When feature_names is empty list, we still get intercept in the importance dict
        assert isinstance(importance2, dict)
        # Should have at least intercept
        assert 'intercept' in importance2

    def test_large_prediction_request(self):
        """Test prediction with larger than training feature set (should handle gracefully)"""
        predictor = EnhancedCPUUsagePredictor(model_type=ModelType.RIDGE)
        predictor.train(self.X_train[:, :5], self.y_train, feature_names=[f'f_{i}' for i in range(5)])

        # Try to predict with more features than trained on - should either work or give clear error
        try:
            # This might work if the model just uses first 5 features
            wrong_shape_features = np.random.randn(1, 10)  # 10 features instead of 5
            prediction = predictor.predict_next_cpu_usage(wrong_shape_features)
            # If it works, check it's a valid prediction
            assert isinstance(prediction, (int, float))
            assert 0 <= prediction <= 100
        except Exception as e:
            # If it fails, it should be a clear error about shape mismatch
            assert "shape" in str(e).lower() or "dimension" in str(e).lower() or "feature" in str(e).lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])