"""
Data pipeline for OpsNexus-ML
Handles ingestion, cleaning, and feature engineering of telemetry data
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpsNexusDataPipeline:
    """
    Pipeline for processing OpsNexus telemetry data
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data pipeline

        Args:
            data_path: Path to JSON file containing telemetry data
        """
        self.data_path = data_path
        self.raw_data = None
        self.df = None
        self.processed_df = None

    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load telemetry data from JSON file

        Args:
            data_path: Path to JSON file (uses instance path if not provided)

        Returns:
            DataFrame with flattened telemetry data
        """
        if data_path is not None:
            self.data_path = data_path

        if self.data_path is None:
            raise ValueError("No data path provided")

        logger.info(f"Loading data from {self.data_path}")

        with open(self.data_path, 'r') as f:
            self.raw_data = json.load(f)

        # Flatten the nested structure for easier processing
        flattened_records = []
        for record in self.raw_data:
            flat_record = {
                'agent_id': record['agent_id'],
                'timestamp': pd.to_datetime(record['timestamp']),
            }

            # Flatten system metrics
            system = record['metrics']['system']
            flat_record.update({
                'cpu_usage_percent': system['cpu']['usage_percent'],
                'cpu_core_count': system['cpu']['count'],
                'memory_usage_percent': system['memory']['usage_percent'],
                'memory_available_mb': system['memory']['available_mb'],
                'memory_used_mb': system['memory']['used_mb'],
                'disk_read_mbps': system['disk']['read_mbps'],
                'disk_write_mbps': system['disk']['write_mbps'],
                'network_bytes_sent': system['network']['bytes_sent'],
                'network_bytes_recv': system['network']['bytes_recv'],
                'uptime_seconds': system['uptime']['uptime'],
                'process_count': system['processes']['count'],
            })

            # Add per-core CPU data if available
            if 'per_cpu' in system['cpu']:
                for i, usage in enumerate(system['cpu']['per_cpu']):
                    flat_record[f'cpu_core_{i}_usage_percent'] = usage

            flattened_records.append(flat_record)

        self.df = pd.DataFrame(flattened_records)
        logger.info(f"Loaded {len(self.df)} data points")

        return self.df

    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean the telemetry data

        Args:
            df: DataFrame to clean (uses self.df if not provided)

        Returns:
            Cleaned DataFrame
        """
        if df is None:
            df = self.df

        if df is None:
            raise ValueError("No data to clean. Load data first.")

        logger.info("Cleaning data...")

        # Make a copy to avoid modifying original
        cleaned_df = df.copy()

        # Handle missing values - forward fill then backward fill
        cleaned_df = cleaned_df.ffill().bfill()

        # Ensure timestamp is datetime and set as index
        cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'])
        cleaned_df = cleaned_df.set_index('timestamp')

        # Remove duplicates (keep first)
        cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep='first')]

        # Clip values to realistic ranges
        cleaned_df['cpu_usage_percent'] = cleaned_df['cpu_usage_percent'].clip(0, 100)
        cleaned_df['memory_usage_percent'] = cleaned_df['memory_usage_percent'].clip(0, 100)
        cleaned_df['disk_read_mbps'] = cleaned_df['disk_read_mbps'].clip(0, None)
        cleaned_df['disk_write_mbps'] = cleaned_df['disk_write_mbps'].clip(0, None)
        cleaned_df['network_bytes_sent'] = cleaned_df['network_bytes_sent'].clip(0, None)
        cleaned_df['network_bytes_recv'] = cleaned_df['network_bytes_recv'].clip(0, None)
        cleaned_df['process_count'] = cleaned_df['process_count'].clip(1, None)

        logger.info(f"Data cleaning complete. Shape: {cleaned_df.shape}")

        self.processed_df = cleaned_df
        return cleaned_df

    def engineer_features(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Engineer features for machine learning models

        Args:
            df: DataFrame to engineer features for (uses processed data if not provided)

        Returns:
            DataFrame with engineered features
        """
        if df is None:
            df = self.processed_df

        if df is None:
            raise ValueError("No processed data available. Clean data first.")

        logger.info("Engineering features...")

        # Make a copy to avoid modifying original
        featured_df = df.copy()

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

        # Lagged features (predicting future based on past)
        for lag in [1, 2, 3, 6, 12]:  # 10s, 20s, 30s, 1min, 2min lags (assuming 10s intervals)
            featured_df[f'cpu_lag_{lag}'] = featured_df['cpu_usage_percent'].shift(lag)
            featured_df[f'memory_lag_{lag}'] = featured_df['memory_usage_percent'].shift(lag)

        # Rolling statistics
        for window in [3, 6, 12]:  # 30s, 1min, 2min windows
            featured_df[f'cpu_rolling_mean_{window}'] = featured_df['cpu_usage_percent'].rolling(window=window).mean()
            featured_df[f'cpu_rolling_std_{window}'] = featured_df['cpu_usage_percent'].rolling(window=window).std()
            featured_df[f'memory_rolling_mean_{window}'] = featured_df['memory_usage_percent'].rolling(window=window).mean()
            featured_df[f'memory_rolling_std_{window}'] = featured_df['memory_usage_percent'].rolling(window=window).std()

        # Rate of change (derivative-like features)
        featured_df['cpu_diff_1'] = featured_df['cpu_usage_percent'].diff(1)
        featured_df['memory_diff_1'] = featured_df['memory_usage_percent'].diff(1)
        featured_df['cpu_pct_change_1'] = featured_df['cpu_usage_percent'].pct_change(1)
        featured_df['memory_pct_change_1'] = featured_df['memory_usage_percent'].pct_change(1)

        # Interaction features
        featured_df['cpu_memory_interaction'] = featured_df['cpu_usage_percent'] * featured_df['memory_usage_percent'] / 100
        featured_df['load_indicator'] = (featured_df['cpu_usage_percent'] + featured_df['memory_usage_percent']) / 2

        # Fill NaN values created by lagging/differencing
        featured_df = featured_df.bfill().ffill()

        logger.info(f"Feature engineering complete. Features: {list(featured_df.columns)}")

        return featured_df

    def prepare_training_data(self,
                            target_column: str = 'cpu_usage_percent',
                            prediction_horizon: int = 6,  # Predict 6 steps ahead (1 minute for 10s intervals)
                            feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training supervised learning models

        Args:
            target_column: Column to predict
            prediction_horizon: How many steps ahead to predict
            feature_columns: List of columns to use as features (uses all if None)

        Returns:
            Tuple of (X, y) arrays for training
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

        X = featured_df[feature_columns].values
        y = featured_df['target'].values

        logger.info(f"Prepared training data: X shape {X.shape}, y shape {y.shape}")

        return X, y, feature_columns

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
            "basic_stats": self.df.describe().to_dict()
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