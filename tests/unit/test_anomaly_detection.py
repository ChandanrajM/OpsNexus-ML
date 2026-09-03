"""
Unit tests for Isolation Forest Anomaly Detection
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.isolation_forest_detector import IsolationForestDetector


class TestIsolationForestDetector:
    """Test cases for IsolationForestDetector"""

    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 10
        self.X_train = np.random.randn(self.n_samples, self.n_features)
        self.X_test = np.random.randn(20, self.n_features)
        # Add some anomalies to test data
        self.X_test[0] = self.X_test[0] * 10  # Extreme outlier

    def test_initialization(self):
        """Test detector initialization"""
        detector = IsolationForestDetector()
        assert detector.estimator is None
        assert detector.feature_names is None
        assert detector.is_trained is False

    def test_train(self):
        """Test model training"""
        detector = IsolationForestDetector()
        detector.train(self.X_train, contamination=0.1, n_estimators=10, random_state=42)

        assert detector.estimator is not None
        assert detector.estimator.n_estimators == 10
        assert detector.is_trained is True

    def test_train_with_feature_names(self):
        """Test training with feature names"""
        detector = IsolationForestDetector()
        feature_names = [f'feature_{i}' for i in range(self.n_features)]
        detector.train(self.X_train, feature_names=feature_names, n_estimators=10, random_state=42)

        assert detector.feature_names == feature_names

    def test_predict_anomaly_score(self):
        """Test anomaly score prediction"""
        detector = IsolationForestDetector()
        detector.train(self.X_train, n_estimators=10, random_state=42)

        scores = detector.predict_anomaly_score(self.X_test)

        assert scores.shape == (20,)
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)
        # First sample should have higher anomaly score (it's an outlier)
        assert scores[0] > np.mean(scores[1:])

    def test_detect_anomaly(self):
        """Test anomaly detection with threshold"""
        detector = IsolationForestDetector()
        detector.train(self.X_train, contamination=0.1, n_estimators=10, random_state=42)

        # Test with default threshold
        predictions = detector.detect_anomaly(self.X_test)
        assert predictions.shape == (20,)
        assert np.all(np.isin(predictions, [0, 1]))

        # Test with custom threshold
        predictions_high = detector.detect_anomaly(self.X_test, threshold=0.5)
        predictions_low = detector.detect_anomaly(self.X_test, threshold=0.9)
        # Lower threshold = more anomalies detected
        assert np.sum(predictions_high) >= np.sum(predictions_low)

    def test_explain_anomaly(self):
        """Test anomaly explanation"""
        detector = IsolationForestDetector()
        detector.train(self.X_train, n_estimators=10, random_state=42, feature_names=[f'f_{i}' for i in range(self.n_features)])

        explanation = detector.explain_anomaly(self.X_test[:1], sample_idx=0, top_n=5)

        assert explanation is not None
        assert 'anomaly_score' in explanation
        assert 'top_contributing_factors' in explanation
        assert len(explanation['top_contributing_factors']) <= 5
        assert all('feature' in f for f in explanation['top_contributing_factors'])
        assert all('deviation_score' in f for f in explanation['top_contributing_factors'])
        assert all('value' in f for f in explanation['top_contributing_factors'])
        assert all('importance' in f for f in explanation['top_contributing_factors'])

    def test_explain_anomaly_multiple_samples(self):
        """Test explanation for multiple samples (summary)"""
        detector = IsolationForestDetector()
        detector.train(self.X_train, n_estimators=10, random_state=42, feature_names=[f'f_{i}' for i in range(self.n_features)])

        explanation = detector.explain_anomaly(self.X_test[:3])  # No sample_idx = summary

        assert explanation is not None
        assert 'n_samples' in explanation
        assert 'anomaly_score_mean' in explanation
        assert 'feature_importance' in explanation

    def test_get_feature_importance(self):
        """Test feature importance extraction"""
        detector = IsolationForestDetector()
        detector.train(self.X_train, n_estimators=10, random_state=42, feature_names=[f'f_{i}' for i in range(self.n_features)])

        importance = detector.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == self.n_features
        assert all(isinstance(v, (int, float)) for v in importance.values())
        assert all(v >= 0 for v in importance.values())

    def test_save_and_load_model(self):
        """Test model persistence"""
        detector = IsolationForestDetector()
        detector.train(self.X_train, n_estimators=10, random_state=42, feature_names=[f'f_{i}' for i in range(self.n_features)])

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_detector.pkl')
            detector.save_model(model_path)

            assert os.path.exists(model_path)

            # Load the model
            loaded_detector = IsolationForestDetector(model_path=model_path)
            loaded_detector.load_model(model_path)

            assert loaded_detector.estimator is not None
            assert loaded_detector.estimator.n_estimators == 10
            assert loaded_detector.feature_names == detector.feature_names

            # Test that loaded model produces same predictions
            original_scores = detector.predict_anomaly_score(self.X_test)
            loaded_scores = loaded_detector.predict_anomaly_score(self.X_test)
            np.testing.assert_array_almost_equal(original_scores, loaded_scores)

    def test_save_model_without_fit(self):
        """Test saving model without fitting raises error"""
        detector = IsolationForestDetector()
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_detector.pkl')
            with pytest.raises(ValueError, match="Cannot save untrained model"):
                detector.save_model(model_path)

    def test_predict_without_fit(self):
        """Test prediction without fitting raises error"""
        detector = IsolationForestDetector()
        with pytest.raises(ValueError, match="Model has not been trained yet"):
            detector.predict_anomaly_score(self.X_test)

    def test_explain_without_fit(self):
        """Test explanation without fitting returns None"""
        detector = IsolationForestDetector()
        result = detector.explain_anomaly(self.X_test[:1])
        assert result is None

    def test_get_feature_importance_without_fit(self):
        """Test feature importance without fitting returns None"""
        detector = IsolationForestDetector()
        result = detector.get_feature_importance()
        assert result is None

    def test_different_contamination_values(self):
        """Test different contamination values"""
        for contamination in [0.01, 0.05, 0.1, 0.2]:
            detector = IsolationForestDetector()
            detector.train(self.X_train, contamination=contamination, n_estimators=10, random_state=42)
            predictions = detector.detect_anomaly(self.X_test)
            assert predictions.shape == (20,)

    def test_reproducibility(self):
        """Test that same random state produces same results"""
        detector1 = IsolationForestDetector()
        detector1.train(self.X_train, n_estimators=10, random_state=42)
        scores1 = detector1.predict_anomaly_score(self.X_test)

        detector2 = IsolationForestDetector()
        detector2.train(self.X_train, n_estimators=10, random_state=42)
        scores2 = detector2.predict_anomaly_score(self.X_test)

        np.testing.assert_array_almost_equal(scores1, scores2)


class TestIsolationForestEdgeCases:
    """Test edge cases for Isolation Forest"""

    def test_single_feature(self):
        """Test with single feature"""
        X = np.random.randn(100, 1)
        detector = IsolationForestDetector()
        detector.train(X, n_estimators=10, random_state=42)
        scores = detector.predict_anomaly_score(X[:5])
        assert scores.shape == (5,)

    def test_large_number_of_features(self):
        """Test with many features"""
        X = np.random.randn(100, 100)
        detector = IsolationForestDetector()
        detector.train(X, n_estimators=10, random_state=42)
        scores = detector.predict_anomaly_score(X[:5])
        assert scores.shape == (5,)

    def test_constant_features(self):
        """Test with constant features"""
        X = np.ones((100, 5))
        X[:, 0] = np.random.randn(100)  # Only first feature varies
        detector = IsolationForestDetector()
        detector.train(X, n_estimators=10, random_state=42)
        scores = detector.predict_anomaly_score(X[:5])
        assert scores.shape == (5,)

    def test_all_same_values(self):
        """Test with all same values"""
        X = np.ones((100, 5)) * 5.0
        detector = IsolationForestDetector()
        detector.train(X, n_estimators=10, random_state=42)
        scores = detector.predict_anomaly_score(X[:5])
        assert scores.shape == (5,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])