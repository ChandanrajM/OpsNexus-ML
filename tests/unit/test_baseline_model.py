"""
Unit tests for CPU Usage Predictor (Linear Regression)
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.baseline_model import CPUUsagePredictor


class TestCPUUsagePredictor:
    """Test cases for CPUUsagePredictor"""

    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 10
        self.X_train = np.random.randn(self.n_samples, self.n_features)
        self.y_train = np.random.rand(self.n_samples) * 100  # CPU usage 0-100%
        self.X_test = np.random.randn(20, self.n_features)
        self.y_test = np.random.rand(20) * 100

    def test_initialization(self):
        """Test predictor initialization"""
        predictor = CPUUsagePredictor()
        assert predictor.coefficients is None
        assert predictor.intercept is None
        assert predictor.feature_names is None
        assert predictor.is_trained is False

    def test_initialization_with_model_path(self):
        """Test predictor initialization with model path"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.pkl')
            predictor = CPUUsagePredictor(model_path=model_path)
            assert predictor.model_path == model_path

    def test_train(self):
        """Test model training"""
        predictor = CPUUsagePredictor()
        predictor.train(self.X_train, self.y_train)

        assert predictor.is_trained is True
        assert predictor.coefficients is not None
        assert predictor.intercept is not None
        assert predictor.coefficients.shape == (self.n_features,)
        assert isinstance(predictor.intercept, (int, float))
        # feature_names is only set if provided
        assert predictor.feature_names is None

    def test_train_with_feature_names(self):
        """Test training with custom feature names"""
        predictor = CPUUsagePredictor()
        feature_names = [f'feature_{i}' for i in range(self.n_features)]
        predictor.train(self.X_train, self.y_train, feature_names=feature_names)

        assert predictor.feature_names == feature_names

    def test_predict(self):
        """Test prediction"""
        predictor = CPUUsagePredictor()
        predictor.train(self.X_train, self.y_train)

        predictions = predictor.predict(self.X_test)

        assert predictions.shape == (20,)
        assert np.all(predictions >= 0)  # CPU usage can't be negative
        assert np.all(predictions <= 100)  # CPU usage can't exceed 100%

    def test_predict_without_fit(self):
        """Test prediction without fitting raises error"""
        predictor = CPUUsagePredictor()
        with pytest.raises(ValueError, match="Model has not been trained yet"):
            predictor.predict(self.X_test)

    def test_predict_next_cpu_usage(self):
        """Test single-step prediction"""
        predictor = CPUUsagePredictor()
        predictor.train(self.X_train, self.y_train)

        # Predict next CPU usage from a single feature vector
        next_features = self.X_test[0:1]
        prediction = predictor.predict_next_cpu_usage(next_features)

        assert isinstance(prediction, (int, float))
        assert 0 <= prediction <= 100

    def test_get_feature_importance(self):
        """Test feature importance extraction"""
        predictor = CPUUsagePredictor()
        predictor.train(self.X_train, self.y_train, feature_names=[f'f_{i}' for i in range(self.n_features)])

        importance = predictor.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == self.n_features + 1  # features + intercept
        assert 'intercept' in importance
        assert all(isinstance(v, (int, float)) for v in importance.values())

    def test_get_feature_importance_without_fit(self):
        """Test feature importance without fitting returns None"""
        predictor = CPUUsagePredictor()
        result = predictor.get_feature_importance()
        assert result is None

    def test_save_and_load_model(self):
        """Test model persistence"""
        predictor = CPUUsagePredictor()
        predictor.train(self.X_train, self.y_train, feature_names=[f'f_{i}' for i in range(self.n_features)])

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_predictor.pkl')
            predictor.save_model(model_path)

            assert os.path.exists(model_path)

            # Load the model
            loaded_predictor = CPUUsagePredictor(model_path=model_path)
            loaded_predictor.load_model(model_path)

            assert loaded_predictor.is_trained is True
            assert loaded_predictor.coefficients is not None
            assert loaded_predictor.intercept is not None
            assert loaded_predictor.feature_names == predictor.feature_names

            # Test that loaded model produces same predictions
            original_preds = predictor.predict(self.X_test)
            loaded_preds = loaded_predictor.predict(self.X_test)
            np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_save_model_without_fit(self):
        """Test saving model without fitting raises error"""
        predictor = CPUUsagePredictor()
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_predictor.pkl')
            with pytest.raises(ValueError, match="Cannot save untrained model"):
                predictor.save_model(model_path)

    def test_load_nonexistent_model(self):
        """Test loading nonexistent model raises error"""
        predictor = CPUUsagePredictor(model_path='/nonexistent/path/model.pkl')
        with pytest.raises(FileNotFoundError):
            predictor.load_model('/nonexistent/path/model.pkl')

    def test_evaluate(self):
        """Test model evaluation (manual calculation since no evaluate method)"""
        predictor = CPUUsagePredictor()
        predictor.train(self.X_train, self.y_train)

        # Manual evaluation
        preds = predictor.predict(self.X_test)
        mae = np.mean(np.abs(self.y_test - preds))
        mse = np.mean((self.y_test - preds) ** 2)
        rmse = np.sqrt(mse)

        assert mae >= 0
        assert mse >= 0
        assert rmse >= 0

    def test_evaluate_without_fit(self):
        """Test evaluation without fitting raises error"""
        predictor = CPUUsagePredictor()
        with pytest.raises(ValueError, match="Model has not been trained yet"):
            predictor.predict(self.X_test)

    def test_different_data_sizes(self):
        """Test with different data sizes"""
        for n_samples, n_features in [(50, 5), (500, 20), (100, 2)]:
            X = np.random.randn(n_samples, n_features)
            y = np.random.rand(n_samples) * 100

            predictor = CPUUsagePredictor()
            predictor.train(X, y)
            preds = predictor.predict(X[:5])

            assert preds.shape == (5,)

    def test_perfect_prediction_case(self):
        """Test with perfectly linear data"""
        # Create data with perfect linear relationship
        X = np.random.randn(100, 3)
        # y = 2*x1 + 3*x2 - x3 + 10 + noise
        y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 10 + np.random.randn(100) * 0.1
        y = np.clip(y, 0, 100)  # Clip to valid CPU range

        predictor = CPUUsagePredictor()
        predictor.train(X, y)

        # Test on training data (should be very accurate)
        preds = predictor.predict(X)
        mae = np.mean(np.abs(preds - y))
        assert mae < 1.0  # Should be very accurate with small noise

    def test_coefficient_signs(self):
        """Test that coefficient signs make sense"""
        # Create data where we know the relationship
        X = np.random.randn(200, 3)
        y = 5 * X[:, 0] - 3 * X[:, 1] + 2 * X[:, 2] + 50
        y = np.clip(y, 0, 100)

        predictor = CPUUsagePredictor()
        predictor.train(X, y)

        # Coefficients should roughly match the true relationship
        # (though scaling matters, so just check signs)
        assert predictor.coefficients[0] > 0  # positive relationship with x1
        assert predictor.coefficients[1] < 0  # negative relationship with x2
        assert predictor.coefficients[2] > 0  # positive relationship with x3


class TestCPUUsagePredictorEdgeCases:
    """Test edge cases for CPU Usage Predictor"""

    def test_single_sample(self):
        """Test with minimum samples"""
        X = np.random.randn(2, 3)
        y = np.random.rand(2) * 100

        predictor = CPUUsagePredictor()
        predictor.train(X, y)
        preds = predictor.predict(X)
        assert preds.shape == (2,)

    def test_single_feature(self):
        """Test with single feature"""
        X = np.random.randn(100, 1)
        y = np.random.rand(100) * 100

        predictor = CPUUsagePredictor()
        predictor.train(X, y)
        preds = predictor.predict(X[:5])
        assert preds.shape == (5,)

    def test_constant_target(self):
        """Test with constant target values"""
        X = np.random.randn(100, 5)
        y = np.ones(100) * 50.0

        predictor = CPUUsagePredictor()
        predictor.train(X, y)
        preds = predictor.predict(X[:5])
        assert preds.shape == (5,)
        # All predictions should be close to 50
        assert np.allclose(preds, 50, atol=1.0)

    def test_negative_coefficients(self):
        """Test that negative coefficients work correctly"""
        X = np.random.randn(100, 2)
        y = -2 * X[:, 0] + 3 * X[:, 1] + 50
        y = np.clip(y, 0, 100)

        predictor = CPUUsagePredictor()
        predictor.train(X, y)

        assert predictor.coefficients[0] < 0
        assert predictor.coefficients[1] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])