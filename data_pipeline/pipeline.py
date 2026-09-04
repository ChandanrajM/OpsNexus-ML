"""
Enhanced data pipeline for OpsNexus-ML
Handles ingestion, cleaning, feature engineering, and optimization of telemetry data
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpsNexusDataPipeline:
    """
    Enhanced pipeline for processing OpsNexus telemetry data
    """

    def __init__(self, data_path: Optional[str] = None, enable_caching: bool = True):
        """
        Initialize the data pipeline

        Args:
            data_path: Path to JSON file containing telemetry data
            enable_caching: Whether to enable data caching for performance
        """
        self.data_path = data_path
        self.enable_caching = enable_caching
        self.raw_data = None
        self.df = None
        self.processed_df = None
        self.featured_df = None
        self._cache = {}
        self._scalers = {}

    def load_data(self, data_path: Optional[str] = None, use_cache: bool = True) -> pd.DataFrame:
        """
        Load telemetry data from JSON file with optional caching

        Args:
            data_path: Path to JSON file (uses instance path if not provided)
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with flattened telemetry data
        """
        if data_path is not None:
            self.data_path = data_path

        if self.data_path is None:
            raise ValueError("No data path provided")

        # Check cache if enabled
        cache_key = f"load_data_{hashlib.md5(self.data_path.encode()).hexdigest()}"
        if self.enable_caching and use_cache and cache_key in self._cache:
            logger.info(f"Loading cached data from {self.data_path}")
            return self._cache[cache_key]

        logger.info(f"Loading data from {self.data_path}")

        try:
            with open(self.data_path, 'r') as f:
                self.raw_data = json.load(f)
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in data file {self.data_path}: {e}")
            raise

        # Flatten the nested structure for easier processing
        flattened_records = []
        processed_count = 0
        skipped_count = 0

        for record in self.raw_data:
            try:
                # Validate required fields
                if 'agent_id' not in record or 'timestamp' not in record or 'metrics' not in record:
                    skipped_count += 1
                    continue

                flat_record = {
                    'agent_id': record['agent_id'],
                    'timestamp': pd.to_datetime(record['timestamp']),
                }

                # Flatten system metrics with error handling
                system = record.get('metrics', {}).get('system', {})

                # CPU metrics
                cpu_data = system.get('cpu', {})
                flat_record.update({
                    'cpu_usage_percent': float(cpu_data.get('usage_percent', 0.0)),
                    'cpu_core_count': int(cpu_data.get('count', 1)),
                })

                # Memory metrics
                memory_data = system.get('memory', {})
                # Calculate memory usage percent from total and available bytes if usage_percent not provided
                if 'usage_percent' in memory_data:
                    memory_usage_percent = float(memory_data.get('usage_percent', 0.0))
                elif 'total_bytes' in memory_data and 'available_bytes' in memory_data:
                    total_bytes = float(memory_data.get('total_bytes', 0.0))
                    available_bytes = float(memory_data.get('available_bytes', 0.0))
                    if total_bytes > 0:
                        memory_usage_percent = ((total_bytes - available_bytes) / total_bytes) * 100
                    else:
                        memory_usage_percent = 0.0
                else:
                    memory_usage_percent = 0.0

                flat_record.update({
                    'memory_usage_percent': memory_usage_percent,
                    'memory_available_mb': float(memory_data.get('available_mb', memory_data.get('available_bytes', 0.0)) / 1024 / 1024),
                    'memory_used_mb': float(memory_data.get('used_mb', memory_data.get('used_bytes', 0.0)) / 1024 / 1024),
                })

                # Disk metrics
                disk_data = system.get('disk', {})
                flat_record.update({
                    'disk_read_mbps': float(disk_data.get('read_mbps', 0.0)),
                    'disk_write_mbps': float(disk_data.get('write_mbps', 0.0)),
                })

                # Network metrics
                network_data = system.get('network', {})
                flat_record.update({
                    'network_bytes_sent': float(network_data.get('bytes_sent', 0.0)),
                    'network_bytes_recv': float(network_data.get('bytes_recv', 0.0)),
                })

                # System metrics
                flat_record.update({
                    'uptime_seconds': float(system.get('uptime', {}).get('uptime', 0.0)),
                    'process_count': int(system.get('processes', {}).get('count', 0)),
                })

                # Add per-core CPU data if available
                if 'per_cpu' in cpu_data:
                    per_cpu = cpu_data['per_cpu']
                    if isinstance(per_cpu, list):
                        for i, usage in enumerate(per_cpu):
                            flat_record[f'cpu_core_{i}_usage_percent'] = float(usage)

                flattened_records.append(flat_record)
                processed_count += 1

            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Skipping invalid record due to: {e}")
                skipped_count += 1
                continue

        if processed_count == 0:
            raise ValueError("No valid data records found in the input file")

        self.df = pd.DataFrame(flattened_records)
        logger.info(f"Loaded {len(self.df)} data points (skipped {skipped_count} invalid records)")

        # Cache the result if enabled
        if self.enable_caching:
            self._cache[cache_key] = self.df.copy()

        return self.df

    def clean_data(self, df: Optional[pd.DataFrame] = None, use_cache: bool = True) -> pd.DataFrame:
        """
        Clean the telemetry data with enhanced error handling and optimization

        Args:
            df: DataFrame to clean (uses self.df if not provided)
            use_cache: Whether to use cached cleaned data if available

        Returns:
            Cleaned DataFrame
        """
        if df is None:
            df = self.df

        if df is None:
            raise ValueError("No data to clean. Load data first.")

        # Check cache if enabled
        cache_key = f"clean_data_{hashlib.md5(str(df.values.tobytes()).encode()).hexdigest()}"
        if self.enable_caching and use_cache and cache_key in self._cache:
            logger.info("Returning cached cleaned data")
            return self._cache[cache_key]

        logger.info("Cleaning data...")

        # Make a copy to avoid modifying original
        cleaned_df = df.copy()

        try:
            # Handle missing values - forward fill then backward fill
            cleaned_df = cleaned_df.ffill().bfill()

            # Ensure timestamp is datetime and set as index
            if 'timestamp' in cleaned_df.columns:
                cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'])
                cleaned_df = cleaned_df.set_index('timestamp')
            elif isinstance(cleaned_df.index, pd.DatetimeIndex):
                # Already has datetime index, ensure it's proper
                cleaned_df.index = pd.to_datetime(cleaned_df.index)
            else:
                logger.warning("No timestamp column or index found for time-based operations")

            # Remove duplicates (keep first)
            initial_count = len(cleaned_df)
            cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep='first')]
            removed_duplicates = initial_count - len(cleaned_df)
            if removed_duplicates > 0:
                logger.info(f"Removed {removed_duplicates} duplicate records")

            # Clip values to realistic ranges with logging
            clip_operations = [
                ('cpu_usage_percent', 0, 100),
                ('memory_usage_percent', 0, 100),
                ('disk_read_mbps', 0, None),
                ('disk_write_mbps', 0, None),
                ('network_bytes_sent', 0, None),
                ('network_bytes_recv', 0, None),
                ('process_count', 1, None)  # At least 1 process
            ]

            for col, min_val, max_val in clip_operations:
                if col in cleaned_df.columns:
                    original_min = cleaned_df[col].min()
                    original_max = cleaned_df[col].max()
                    cleaned_df[col] = cleaned_df[col].clip(min_val, max_val)
                    new_min = cleaned_df[col].min()
                    new_max = cleaned_df[col].max()
                    if original_min != new_min or original_max != new_max:
                        logger.debug(f"Clipped {col}: [{original_min:.2f}, {original_max:.2f}] -> [{new_min:.2f}, {new_max:.2f}]")

            logger.info(f"Data cleaning complete. Shape: {cleaned_df.shape}")

            self.processed_df = cleaned_df

            # Cache the result if enabled
            if self.enable_caching:
                self._cache[cache_key] = cleaned_df.copy()

            return cleaned_df

        except Exception as e:
            logger.error(f"Error during data cleaning: {e}")
            # Return original data if cleaning fails critically
            return df.copy()

    def engineer_features(self, df: Optional[pd.DataFrame] = None, use_cache: bool = True) -> pd.DataFrame:
        """
        Engineer features for machine learning models with performance optimizations

        Args:
            df: DataFrame to engineer features for (uses processed data if not provided)
            use_cache: Whether to use cached featured data if available

        Returns:
            DataFrame with engineered features
        """
        if df is None:
            df = self.processed_df

        if df is None:
            raise ValueError("No processed data available. Clean data first.")

        # Check cache if enabled
        cache_key = f"engineer_features_{hashlib.md5(str(df.values.tobytes()).encode()).hexdigest()}"
        if self.enable_caching and use_cache and cache_key in self._cache:
            logger.info("Returning cached featured data")
            return self._cache[cache_key]

        logger.info("Engineering features...")

        # Make a copy to avoid modifying original
        featured_df = df.copy()

        try:
            # Time-based features
            featured_df['hour_of_day'] = featured_df.index.hour
            featured_df['day_of_week'] = featured_df.index.dayofweek  # Monday=0, Sunday=6
            featured_df['is_weekend'] = (featured_df['day_of_week'] >= 5).astype(int)
            featured_df['is_business_hour'] = ((featured_df['hour_of_day'] >= 8) &
                                              (featured_df['hour_of_day'] <= 18)).astype(int)

            # Cyclical encoding for time features
            featured_df['hour_sin'] = np.sin(2 * np.pi * featured_df['hour_of_day'] / 24)
            featured_df['hour_cos'] = np.cos(2 * np.pi * featured_df['hour_of_day'] / 24)
            featured_df['day_sin'] = np.sin(2 * np.pi * featured_df['day_of_week'] / 7)
            featured_df['day_cos'] = np.cos(2 * np.pi * featured_df['day_of_week'] / 7)

            # Lagged features (predicting future based on past) - optimized
            max_lag = 12
            for lag in [1, 2, 3, 6, 12]:  # 10s, 20s, 30s, 1min, 2min lags (assuming 10s intervals)
                featured_df[f'cpu_lag_{lag}'] = featured_df['cpu_usage_percent'].shift(lag)
                featured_df[f'memory_lag_{lag}'] = featured_df['memory_usage_percent'].shift(lag)

            # Rolling statistics - optimized with min_periods
            for window in [3, 6, 12]:  # 30s, 1min, 2min windows
                featured_df[f'cpu_rolling_mean_{window}'] = featured_df['cpu_usage_percent'].rolling(
                    window=window, min_periods=1).mean()
                featured_df[f'cpu_rolling_std_{window}'] = featured_df['cpu_usage_percent'].rolling(
                    window=window, min_periods=1).std()
                featured_df[f'memory_rolling_mean_{window}'] = featured_df['memory_usage_percent'].rolling(
                    window=window, min_periods=1).mean()
                featured_df[f'memory_rolling_std_{window}'] = featured_df['memory_usage_percent'].rolling(
                    window=window, min_periods=1).std()

            # Rate of change (derivative-like features)
            featured_df['cpu_diff_1'] = featured_df['cpu_usage_percent'].diff(1)
            featured_df['memory_diff_1'] = featured_df['memory_usage_percent'].diff(1)
            featured_df['cpu_pct_change_1'] = featured_df['cpu_usage_percent'].pct_change(1)
            featured_df['memory_pct_change_1'] = featured_df['memory_usage_percent'].pct_change(1)

            # Interaction features
            featured_df['cpu_memory_interaction'] = featured_df['cpu_usage_percent'] * featured_df['memory_usage_percent'] / 100
            featured_df['load_indicator'] = (featured_df['cpu_usage_percent'] + featured_df['memory_usage_percent']) / 2

            # Advanced features
            # Exponential moving averages for trend detection
            featured_df['cpu_ema_12'] = featured_df['cpu_usage_percent'].ewm(span=12).mean()
            featured_df['memory_ema_12'] = featured_df['memory_usage_percent'].ewm(span=12).mean()

            # Momentum indicators
            featured_df['cpu_momentum'] = featured_df['cpu_usage_percent'] - featured_df['cpu_lag_12']
            featured_df['memory_momentum'] = featured_df['memory_usage_percent'] - featured_df['memory_lag_12']

            # Volatility indicators
            featured_df['cpu_volatility'] = featured_df['cpu_usage_percent'].rolling(window=12).std()
            featured_df['memory_volatility'] = featured_df['memory_usage_percent'].rolling(window=12).std()

            # Fill NaN values created by lagging/differencing
            featured_df = featured_df.bfill().ffill()

            logger.info(f"Feature engineering complete. Features: {list(featured_df.columns)}")

            # Cache the result if enabled
            if self.enable_caching:
                self._cache[cache_key] = featured_df.copy()

            return featured_df

        except Exception as e:
            logger.error(f"Error during feature engineering: {e}")
            # Return original data if feature engineering fails critically
            return df.copy()

    def prepare_training_data(self,
                            target_column: str = 'cpu_usage_percent',
                            prediction_horizon: int = 6,  # Predict 6 steps ahead (1 minute for 10s intervals)
                            feature_columns: Optional[List[str]] = None,
                            scale_features: bool = True,
                            feature_scale_type: str = 'standard') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for training supervised learning models with feature scaling options

        Args:
            target_column: Column to predict
            prediction_horizon: How many steps ahead to predict
            feature_columns: List of columns to use as features (uses all if None)
            scale_features: Whether to scale features using standardization or normalization
            feature_scale_type: Type of scaling ('standard' for StandardScaler, 'minmax' for MinMaxScaler)

        Returns:
            Tuple of (X, y, feature_names) arrays for training
        """
        if self.processed_df is None:
            raise ValueError("No processed data available. Run clean_data() first.")

        logger.info(f"Preparing training data for {target_column} with {prediction_horizon} step horizon...")

        # Engineer features
        featured_df = self.engineer_features()

        # Define target variable (future value)
        featured_df['target'] = featured_df[target_column].shift(-prediction_horizon)

        # Remove rows where target is NaN (at the end)
        featured_df = featured_df.dropna(subset=['target'])

        # Select features
        if feature_columns is None:
            # Exclude non-feature columns
            exclude_cols = ['target', 'agent_id']
            feature_columns = [col for col in featured_df.columns if col not in exclude_cols]
        else:
            # Validate that requested columns exist
            missing_cols = [col for col in feature_columns if col not in featured_df.columns]
            if missing_cols:
                raise ValueError(f"Feature columns not found in data: {missing_cols}")

        X = featured_df[feature_columns].values
        y = featured_df['target'].values

        # Scale features if requested
        if scale_features and len(feature_columns) > 0:
            try:
                if feature_scale_type == 'standard':
                    scaler = StandardScaler()
                elif feature_scale_type == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    logger.warning(f"Unknown scale type {feature_scale_type}, using standard scaling")
                    scaler = StandardScaler()

                X_scaled = scaler.fit_transform(X)

                # Store scaler for potential reuse
                scaler_key = f"scaler_{feature_scale_type}_{hashlib.md5(str(feature_columns).encode()).hexdigest()}"
                self._scalers[scaler_key] = scaler

                X = X_scaled
                logger.info(f"Features scaled using {feature_scale_type} scaling")

            except Exception as e:
                logger.warning(f"Feature scaling failed: {e}. Proceeding with unscaled features.")

        logger.info(f"Prepared training data: X shape {X.shape}, y shape {y.shape}, features: {len(feature_columns)}")

        return X, y, feature_columns

    def get_feature_scaler(self, feature_columns: List[str], scale_type: str = 'standard'):
        """
        Get a fitted feature scaler for consistent transformation

        Args:
            feature_columns: List of feature column names
            scale_type: Type of scaler to retrieve

        Returns:
            Fitted scaler object or None if not found
        """
        scaler_key = f"scaler_{scale_type}_{hashlib.md5(str(feature_columns).encode()).hexdigest()}"
        return self._scalers.get(scaler_key)

    def get_data_info(self) -> Dict:
        """
        Get information about the loaded data

        Returns:
            Dictionary with data statistics
        """
        if self.df is None:
            return {"error": "No data loaded"}

        info = {
            "total_records": len(self.df),
            "date_range": {
                "start": self.df['timestamp'].min().isoformat() if 'timestamp' in self.df.columns else None,
                "end": self.df['timestamp'].max().isoformat() if 'timestamp' in self.df.columns else None
            },
            "columns": list(self.df.columns),
            "missing_values": self.df.isnull().sum().to_dict(),
            "basic_stats": self.df.describe().to_dict(),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024 / 1024
        }

        # Add data quality metrics
        if len(self.df) > 0:
            info["quality_metrics"] = {
                "complete_records": self.df.dropna().shape[0],
                "completeness_percentage": (self.df.dropna().shape[0] / len(self.df)) * 100,
                "duplicate_records": self.df.duplicated().sum(),
                "memory_efficiency": "high" if info["memory_usage_mb"] < 10 else "medium" if info["memory_usage_mb"] < 100 else "low"
            }

        return info

def main():
    """
    Example usage of the data pipeline
    """
    # Initialize pipeline
    pipeline = OpsNexusDataPipeline()

    # Load synthetic data
    data_path = "/home/chandanraj-m/OpsNexus-ML/data_pipeline/synthetic_telemetry_sample.json"
    df = pipeline.load_data(data_path)

    # Show data info
    info = pipeline.get_data_info()
    print("Data Info:")
    print(json.dumps(info, indent=2, default=str))

    # Clean data
    cleaned_df = pipeline.clean_data()
    print(f"\nCleaned data shape: {cleaned_df.shape}")

    # Engineer features
    featured_df = pipeline.engineer_features()
    print(f"Featured data shape: {featured_df.shape}")
    print(f"Features: {list(featured_df.columns)}")

    # Prepare training data
    X, y, feature_cols = pipeline.prepare_training_data(
        target_column='cpu_usage_percent',
        prediction_horizon=6  # Predict 1 minute ahead
    )
    print(f"\nTraining data prepared:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Number of features: {len(feature_cols)}")

if __name__ == "__main__":
    main()