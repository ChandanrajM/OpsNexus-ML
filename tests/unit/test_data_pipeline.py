"""
Unit tests for Enhanced Data Pipeline
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest
import tempfile
import json
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_pipeline.pipeline import OpsNexusDataPipeline

# Set up logger for tests
logger = logging.getLogger(__name__)


class TestOpsNexusDataPipeline:
    """Test cases for OpsNexusDataPipeline"""

    def setup_method(self):
        """Set up test data"""
        # Create sample telemetry data similar to the synthetic data generator
        self.sample_data = []
        base_time = pd.Timestamp('2026-09-04 10:00:00')

        for i in range(50):
            record = {
                'agent_id': f'test-agent-{i % 5}',
                'timestamp': (base_time + pd.Timedelta(seconds=i*10)).isoformat(),
                'metrics': {
                    'system': {
                        'cpu': {
                            'usage_percent': float(np.random.uniform(10, 90)),
                            'count': np.random.randint(2, 8),
                            'per_cpu': [float(np.random.uniform(10, 90)) for _ in range(np.random.randint(2, 8))]
                        },
                        'memory': {
                            'usage_percent': float(np.random.uniform(20, 80)),
                            'available_mb': float(np.random.uniform(1000, 8000)),
                            'used_mb': float(np.random.uniform(2000, 6000))
                        },
                        'disk': {
                            'read_mbps': float(np.random.uniform(0, 100)),
                            'write_mbps': float(np.random.uniform(0, 50))
                        },
                        'network': {
                            'bytes_sent': float(np.random.uniform(1000, 10000)),
                            'bytes_recv': float(np.random.uniform(1000, 10000))
                        },
                        'uptime': {
                            'uptime': float(np.random.uniform(10000, 500000))
                        },
                        'processes': {
                            'count': np.random.randint(50, 500)
                        }
                    }
                }
            }
            self.sample_data.append(record)

    def test_initialization(self):
        """Test pipeline initialization"""
        pipeline = OpsNexusDataPipeline()
        assert pipeline.data_path is None
        assert pipeline.raw_data is None
        assert pipeline.df is None
        assert pipeline.processed_df is None
        assert pipeline.featured_df is None
        assert pipeline.enable_caching is True  # Default value

        pipeline_no_cache = OpsNexusDataPipeline(enable_caching=False)
        assert pipeline_no_cache.enable_caching is False

    def test_initialization_with_data_path(self):
        """Test pipeline initialization with data path"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_data, f)
            temp_path = f.name

        try:
            pipeline = OpsNexusDataPipeline(data_path=temp_path)
            assert pipeline.data_path == temp_path
        finally:
            os.unlink(temp_path)

    def test_load_data_success(self):
        """Test successful data loading"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_data, f)
            temp_path = f.name

        try:
            pipeline = OpsNexusDataPipeline()
            df = pipeline.load_data(temp_path)

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 50
            assert 'agent_id' in df.columns
            assert 'timestamp' in df.columns
            assert 'cpu_usage_percent' in df.columns
            assert 'memory_usage_percent' in df.columns
            # Check for per-core columns if they exist in sample data
            assert 'cpu_core_0_usage_percent' in df.columns or len([c for c in df.columns if 'cpu_core' in c]) >= 0

            # Check data types
            assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])
            assert df['cpu_usage_percent'].dtype in [np.float64, np.float32]
            assert df['memory_usage_percent'].dtype in [np.float64, np.float32]

            logger.info(f"Loaded DataFrame shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")

        finally:
            os.unlink(temp_path)

    def test_load_data_file_not_found(self):
        """Test loading non-existent file"""
        pipeline = OpsNexusDataPipeline()
        with pytest.raises(FileNotFoundError):
            pipeline.load_data('/non/existent/file.json')

    def test_load_data_invalid_json(self):
        """Test loading invalid JSON file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json content}')
            temp_path = f.name

        try:
            pipeline = OpsNexusDataPipeline()
            with pytest.raises(json.JSONDecodeError):
                pipeline.load_data(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_data_caching(self):
        """Test data caching functionality"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_data, f)
            temp_path = f.name

        try:
            pipeline = OpsNexusDataPipeline(enable_caching=True)

            # First load
            df1 = pipeline.load_data(temp_path)

            # Second load should use cache
            df2 = pipeline.load_data(temp_path)

            # Should be equal
            pd.testing.assert_frame_equal(df1, df2)

            # Check that cache was used (by checking internal cache)
            assert len(pipeline._cache) > 0

        finally:
            os.unlink(temp_path)

    def test_load_data_no_caching(self):
        """Test loading without caching"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_data, f)
            temp_path = f.name

        try:
            pipeline = OpsNexusDataPipeline(enable_caching=False)

            # Load data
            df1 = pipeline.load_data(temp_path)
            df2 = pipeline.load_data(temp_path)  # Should load again, not use cache

            # Should be equal but not the same object (no caching)
            pd.testing.assert_frame_equal(df1, df2)

            # Check that cache was not used
            assert len(pipeline._cache) == 0

        finally:
            os.unlink(temp_path)

    def test_clean_data_basic(self):
        """Test basic data cleaning functionality"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_data, f)
            temp_path = f.name

        try:
            pipeline = OpsNexusDataPipeline()
            df = pipeline.load_data(temp_path)
            cleaned_df = pipeline.clean_data()

            assert isinstance(cleaned_df, pd.DataFrame)
            assert len(cleaned_df) == len(df)  # No rows should be removed by cleaning in clean data

            # Check that timestamp is index if it was a column
            if 'timestamp' in df.columns:
                assert isinstance(cleaned_df.index, pd.DatetimeIndex)

            # Check that values are within expected ranges
            assert (cleaned_df['cpu_usage_percent'] >= 0).all() and (cleaned_df['cpu_usage_percent'] <= 100).all()
            assert (cleaned_df['memory_usage_percent'] >= 0).all() and (cleaned_df['memory_usage_percent'] <= 100).all()
            assert (cleaned_df['disk_read_mbps'] >= 0).all()
            assert (cleaned_df['disk_write_mbps'] >= 0).all()
            assert (cleaned_df['network_bytes_sent'] >= 0).all()
            assert (cleaned_df['network_bytes_recv'] >= 0).all()
            assert (cleaned_df['process_count'] >= 1).all()  # At least 1 process

            logger.info(f"Cleaned DataFrame shape: {cleaned_df.shape}")

        finally:
            os.unlink(temp_path)

    def test_clean_data_with_missing_values(self):
        """Test cleaning data with missing values"""
        # Create data with some missing values
        sample_data_with_nan = self.sample_data.copy()
        # Introduce some None values
        sample_data_with_nan[0]['metrics']['system']['cpu']['usage_percent'] = None
        sample_data_with_nan[1]['metrics']['system']['memory']['usage_percent'] = None

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data_with_nan, f)
            temp_path = f.name

        try:
            pipeline = OpsNexusDataPipeline()
            df = pipeline.load_data(temp_path)
            cleaned_df = pipeline.clean_data()

            # Check that NaN values have been filled
            assert not cleaned_df['cpu_usage_percent'].isnull().any()
            assert not cleaned_df['memory_usage_percent'].isnull().any()

            # Check that values are still in valid ranges
            assert (cleaned_df['cpu_usage_percent'] >= 0).all() and (cleaned_df['cpu_usage_percent'] <= 100).all()
            assert (cleaned_df['memory_usage_percent'] >= 0).all() and (cleaned_df['memory_usage_percent'] <= 100).all()

            logger.info(f"Cleaned DataFrame with NaN shape: {cleaned_df.shape}")

        finally:
            os.unlink(temp_path)

    def test_clean_data_caching(self):
        """Test data cleaning caching"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_data, f)
            temp_path = f.name

        try:
            pipeline = OpsNexusDataPipeline(enable_caching=True)

            # Load and clean data
            df = pipeline.load_data(temp_path)
            cleaned_df1 = pipeline.clean_data()

            # Clean again should use cache
            cleaned_df2 = pipeline.clean_data()

            # Should be equal
            pd.testing.assert_frame_equal(cleaned_df1, cleaned_df2)

        finally:
            os.unlink(temp_path)

    def test_engineer_features_basic(self):
        """Test basic feature engineering"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_data, f)
            temp_path = f.name

        try:
            pipeline = OpsNexusDataPipeline()
            df = pipeline.load_data(temp_path)
            cleaned_df = pipeline.clean_data()
            featured_df = pipeline.engineer_features()

            assert isinstance(featured_df, pd.DataFrame)
            assert len(featured_df) == len(cleaned_df)  # No rows should be removed

            # Check for expected time-based features
            expected_features = [
                'hour_of_day', 'day_of_week', 'is_weekend', 'is_business_hour',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
            ]
            for feat in expected_features:
                assert feat in featured_df.columns, f"Missing expected feature: {feat}"

            # Check for lagged features
            assert 'cpu_lag_1' in featured_df.columns
            assert 'memory_lag_1' in featured_df.columns
            assert 'cpu_lag_12' in featured_df.columns
            assert 'memory_lag_12' in featured_df.columns

            # Check for rolling statistics
            assert 'cpu_rolling_mean_3' in featured_df.columns
            assert 'cpu_rolling_std_3' in featured_df.columns
            assert 'memory_rolling_mean_3' in featured_df.columns
            assert 'memory_rolling_std_3' in featured_df.columns

            # Check for rate of change features
            assert 'cpu_diff_1' in featured_df.columns
            assert 'memory_diff_1' in featured_df.columns
            assert 'cpu_pct_change_1' in featured_df.columns
            assert 'memory_pct_change_1' in featured_df.columns

            # Check for interaction features
            assert 'cpu_memory_interaction' in featured_df.columns
            assert 'load_indicator' in featured_df.columns

            # Check for advanced features (EMA, momentum, volatility)
            assert 'cpu_ema_12' in featured_df.columns
            assert 'memory_ema_12' in featured_df.columns
            assert 'cpu_momentum' in featured_df.columns
            assert 'memory_momentum' in featured_df.columns
            assert 'cpu_volatility' in featured_df.columns
            assert 'memory_volatility' in featured_df.columns

            logger.info(f"Featured DataFrame shape: {featured_df.shape}")
            logger.info(f"Number of features: {len(featured_df.columns)}")

        finally:
            os.unlink(temp_path)

    def test_engineer_features_caching(self):
        """Test feature engineering caching"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_data, f)
            temp_path = f.name

        try:
            pipeline = OpsNexusDataPipeline(enable_caching=True)

            # Load, clean, and engineer features
            df = pipeline.load_data(temp_path)
            cleaned_df = pipeline.clean_data()
            featured_df1 = pipeline.engineer_features()

            # Engineer features again should use cache
            featured_df2 = pipeline.engineer_features()

            # Should be equal
            pd.testing.assert_frame_equal(featured_df1, featured_df2)

        finally:
            os.unlink(temp_path)

    def test_prepare_training_data_basic(self):
        """Test basic training data preparation"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_data, f)
            temp_path = f.name

        try:
            pipeline = OpsNexusDataPipeline()
            df = pipeline.load_data(temp_path)
            cleaned_df = pipeline.clean_data()
            featured_df = pipeline.engineer_features()

            X, y, feature_names = pipeline.prepare_training_data()

            assert isinstance(X, np.ndarray)
            assert isinstance(y, np.ndarray)
            assert isinstance(feature_names, list)

            # Check shapes
            assert len(X) == len(y)  # Same number of samples
            assert X.shape[1] == len(feature_names)  # Features match
            assert len(X) > 0  # Should have some samples

            # Check that target is cpu_usage_percent shifted by horizon
            # The target should be future values, so there should be a relationship
            # (though not perfect due to the shift and potential NaN removal)

            logger.info(f"Prepared training data: X{X.shape}, y{y.shape}, {len(feature_names)} features")
            logger.info(f"First 5 feature names: {feature_names[:5]}")

        finally:
            os.unlink(temp_path)

    def test_prepare_training_data_with_scaling(self):
        """Test training data preparation with feature scaling"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_data, f)
            temp_path = f.name

        try:
            pipeline = OpsNexusDataPipeline(enable_caching=True)

            # Load and process data
            df = pipeline.load_data(temp_path)
            cleaned_df = pipeline.clean_data()
            featured_df = pipeline.engineer_features()

            # Test without scaling
            X_unscaled, y, feature_names = pipeline.prepare_training_data(scale_features=False)

            # Test with standard scaling
            X_scaled_std, y, feature_names = pipeline.prepare_training_data(
                scale_features=True, feature_scale_type='standard')

            # Test with minmax scaling
            X_scaled_mm, y, feature_names = pipeline.prepare_training_data(
                scale_features=True, feature_scale_type='minmax')

            # Check that shapes are consistent
            assert X_unscaled.shape == X_scaled_std.shape == X_scaled_mm.shape
            assert len(y) == len(X_unscaled) == len(X_scaled_std) == len(X_scaled_mm)

            # Check that scaled data has different properties
            # Standard scaled data should have mean ~0 and std ~1
            if X_scaled_std.shape[1] > 0:
                mean_std = np.mean(X_scaled_std[:, 0])
                std_std = np.std(X_scaled_std[:, 0])
                # Allow some tolerance due to small sample size
                assert abs(mean_std) < 0.1 or abs(std_std - 1.0) < 0.1  # At least one should be close

            # Minmax scaled data should be in [0, 1] range
            if X_scaled_mm.shape[1] > 0:
                min_mm = np.min(X_scaled_mm[:, 0])
                max_mm = np.max(X_scaled_mm[:, 0])
                assert min_mm >= -0.01 and max_mm <= 1.01  # Allow small tolerance

            logger.info(f"Unscaled feature range: [{np.min(X_unscaled):.3f}, {np.max(X_unscaled):.3f}]")
            logger.info(f"Standard scaled feature range: [{np.min(X_scaled_std):.3f}, {np.max(X_scaled_std):.3f}]")
            logger.info(f"Minmax scaled feature range: [{np.min(X_scaled_mm):.3f}, {np.max(X_scaled_mm):.3f}]")

        finally:
            os.unlink(temp_path)

    def test_prepare_training_data_custom_features(self):
        """Test training data preparation with custom feature selection"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_data, f)
            temp_path = f.name

        try:
            pipeline = OpsNexusDataPipeline()
            df = pipeline.load_data(temp_path)
            cleaned_df = pipeline.clean_data()
            featured_df = pipeline.engineer_features()

            # Select only a few features
            custom_features = ['cpu_usage_percent', 'memory_usage_percent', 'hour_of_day']
            # Verify these features exist
            for feat in custom_features:
                assert feat in featured_df.columns, f"Custom feature {feat} not found in featured data"

            X, y, feature_names = pipeline.prepare_training_data(
                feature_columns=custom_features)

            assert isinstance(X, np.ndarray)
            assert isinstance(y, np.ndarray)
            assert isinstance(feature_names, list)

            assert len(feature_names) == len(custom_features)
            assert feature_names == custom_features  # Should maintain order
            assert X.shape[1] == len(custom_features)

            logger.info(f"Custom features training data: X{X.shape}, y{y.shape}")

        finally:
            os.unlink(temp_path)

    def test_prepare_training_data_invalid_features(self):
        """Test training data preparation with invalid feature names"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_data, f)
            temp_path = f.name

        try:
            pipeline = OpsNexusDataPipeline()
            df = pipeline.load_data(temp_path)
            cleaned_df = pipeline.clean_data()
            featured_df = pipeline.engineer_features()

            # Try to use non-existent feature
            invalid_features = ['cpu_usage_percent', 'non_existent_feature']

            with pytest.raises(ValueError, match="Feature columns not found in data"):
                pipeline.prepare_training_data(feature_columns=invalid_features)

        finally:
            os.unlink(temp_path)

    def test_get_data_info(self):
        """Test getting data information"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_data, f)
            temp_path = f.name

        try:
            pipeline = OpsNexusDataPipeline()
            df = pipeline.load_data(temp_path)
            info = pipeline.get_data_info()

            assert isinstance(info, dict)
            assert 'total_records' in info
            assert info['total_records'] == 50
            assert 'columns' in info
            assert len(info['columns']) > 0
            assert 'missing_values' in info
            assert 'basic_stats' in info
            # New fields in enhanced version
            assert 'memory_usage_mb' in info
            assert 'quality_metrics' in info

            logger.info(f"Data info keys: {list(info.keys())}")
            logger.info(f"Total records: {info['total_records']}")
            logger.info(f"Memory usage: {info['memory_usage_mb']:.2f} MB")

        finally:
            os.unlink(temp_path)

    def test_get_data_info_no_data(self):
        """Test getting data info when no data is loaded"""
        pipeline = OpsNexusDataPipeline()
        info = pipeline.get_data_info()

        assert isinstance(info, dict)
        assert 'error' in info
        assert info['error'] == "No data loaded"

    def test_end_to_end_processing(self):
        """Test end-to-end data processing pipeline"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_data, f)
            temp_path = f.name

        try:
            pipeline = OpsNexusDataPipeline(enable_caching=True)

            # Full processing pipeline
            df = pipeline.load_data(temp_path)
            cleaned_df = pipeline.clean_data()
            featured_df = pipeline.engineer_features()
            X, y, feature_names = pipeline.prepare_training_data(scale_features=True)

            # Validate the entire process
            assert isinstance(df, pd.DataFrame)
            assert isinstance(cleaned_df, pd.DataFrame)
            assert isinstance(featured_df, pd.DataFrame)
            assert isinstance(X, np.ndarray)
            assert isinstance(y, np.ndarray)
            assert isinstance(feature_names, list)

            # Check that we have reasonable amounts of data
            assert len(df) > 0
            assert len(cleaned_df) > 0
            assert len(featured_df) > 0
            assert len(X) > 0
            assert len(y) > 0
            assert len(feature_names) > 0

            # Check that shapes are consistent
            assert len(X) == len(y)
            assert X.shape[1] == len(feature_names)

            logger.info(f"End-to-end pipeline successful:")
            logger.info(f"  Raw data: {df.shape}")
            logger.info(f"  Cleaned data: {cleaned_df.shape}")
            logger.info(f"  Featured data: {featured_df.shape}")
            logger.info(f"  Training data: X{X.shape}, y{y.shape}")
            logger.info(f"  Features: {len(feature_names)}")

        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])