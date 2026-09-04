"""
Enhanced visualization suite for OpsNexus-ML
Provides comprehensive visualizations for model performance, anomaly detection, and system insights
"""
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd

# Add project directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_pipeline'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

from pipeline import OpsNexusDataPipeline
from baseline_model import CPUUsagePredictor
from models.isolation_forest_detector import IsolationForestDetector
from models.enhanced_model import EnhancedCPUUsagePredictor, ModelType

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class OpsNexusMLVisualizer:
    """
    Comprehensive visualization suite for OpsNexus-ML
    """

    def __init__(self, viz_dir: str = None):
        """
        Initialize the visualizer

        Args:
            viz_dir: Directory to save visualizations (defaults to ./visualization/)
        """
        if viz_dir is None:
            viz_dir = os.path.join(os.path.dirname(__file__))

        self.viz_dir = viz_dir
        os.makedirs(self.viz_dir, exist_ok=True)

        # Initialize data pipeline and models
        self.data_pipeline = OpsNexusDataPipeline()
        self.baseline_predictor = None
        self.enhanced_predictor = None
        self.anomaly_detector = None
        self.feature_names = None

    def load_and_prepare_data(self, data_path: str = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load and prepare data for visualization

        Returns:
            Tuple of (X, y, feature_names)
        """
        if data_path is None:
            data_path = "/home/chandanraj-m/OpsNexus-ML/data_pipeline/synthetic_telemetry_sample.json"

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Load and process data
        df = self.data_pipeline.load_data(data_path)
        cleaned_df = self.data_pipeline.clean_data()
        featured_df = self.data_pipeline.engineer_features()

        # Prepare data for prediction
        X, y, feature_names = self.data_pipeline.prepare_training_data(
            target_column='cpu_usage_percent',
            prediction_horizon=6  # Predict 1 minute ahead
        )

        self.feature_names = feature_names
        return X, y, feature_names

    def load_models(self):
        """Load pre-trained models"""
        # Load baseline model
        baseline_model_path = "/home/chandanraj-m/OpsNexus-ML/models/cpu_predictor.pkl"
        if os.path.exists(baseline_model_path):
            self.baseline_predictor = CPUUsagePredictor(model_path=baseline_model_path)
            self.baseline_predictor.load_model(baseline_model_path)
            print(f"Loaded baseline model from {baseline_model_path}")

        # Load enhanced model (try to load, will fallback to training if needed)
        enhanced_model_path = "/home/chandanraj-m/OpsNexus-ML/models/enhanced_cpu_predictor.pkl"
        if os.path.exists(enhanced_model_path):
            self.enhanced_predictor = EnhancedCPUUsagePredictor(model_path=enhanced_model_path)
            self.enhanced_predictor.load_model(enhanced_model_path)
            print(f"Loaded enhanced model from {enhanced_model_path}")

        # Load anomaly detector
        anomaly_model_path = "/home/chandanraj-m/OpsNexus-ML/models/isolation_forest_detector.pkl"
        if os.path.exists(anomaly_model_path):
            self.anomaly_detector = IsolationForestDetector(model_path=anomaly_model_path)
            self.anomaly_detector.load_model(anomaly_model_path)
            print(f"Loaded anomaly detector from {anomaly_model_path}")

    def plot_model_comparison(self) -> bool:
        """
        Create a comprehensive model comparison visualization

        Returns:
            Success status
        """
        print("Creating model comparison visualization...")

        try:
            # Load data
            X, y, feature_names = self.load_and_prepare_data()

            # Load or train models
            if self.baseline_predictor is None:
                print("Training baseline model for comparison...")
                self.baseline_predictor = CPUUsagePredictor()
                baseline_metrics = self.baseline_predictor.train(X, y, feature_names=feature_names)

            if self.enhanced_predictor is None:
                print("Training enhanced models for comparison...")
                # Train multiple enhanced models and pick the best
                comparison_result = EnhancedCPUUsagePredictor.compare_models(
                    X, y, feature_names=feature_names
                )
                self.enhanced_predictor = comparison_result['best_model']
                print(f"Selected best enhanced model: {comparison_result['best_model_type']}")

            # Make predictions with both models
            y_pred_baseline = self.baseline_predictor.predict(X)
            y_pred_enhanced = self.enhanced_predictor.predict(X)

            # Calculate metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            baseline_mae = mean_absolute_error(y, y_pred_baseline)
            baseline_rmse = np.sqrt(mean_squared_error(y, y_pred_baseline))
            baseline_r2 = r2_score(y, y_pred_baseline)

            enhanced_mae = mean_absolute_error(y, y_pred_enhanced)
            enhanced_rmse = np.sqrt(mean_squared_error(y, y_pred_enhanced))
            enhanced_r2 = r2_score(y, y_pred_enhanced)

            # Create the visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

            # Plot 1: Prediction accuracy comparison
            ax1 = axes[0, 0]
            ax1.scatter(y, y_pred_baseline, alpha=0.6, s=20, label=f'Baseline (MAE: {baseline_mae:.3f})', color='blue')
            ax1.scatter(y, y_pred_enhanced, alpha=0.6, s=20, label=f'Enhanced (MAE: {enhanced_mae:.3f})', color='red')

            # Perfect prediction line
            min_val = min(min(y), min(y_pred_baseline), min(y_pred_enhanced))
            max_val = max(max(y), max(y_pred_baseline), max(y_pred_enhanced))
            ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')

            ax1.set_xlabel('Actual CPU Usage (%)')
            ax1.set_ylabel('Predicted CPU Usage (%)')
            ax1.set_title('Actual vs Predicted Values')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')

            # Plot 2: Residuals comparison
            ax2 = axes[0, 1]
            residuals_baseline = y - y_pred_baseline
            residuals_enhanced = y - y_pred_enhanced

            ax2.scatter(y_pred_baseline, residuals_baseline, alpha=0.6, s=20, label='Baseline', color='blue')
            ax2.scatter(y_pred_enhanced, residuals_enhanced, alpha=0.6, s=20, label='Enhanced', color='red')
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.8)

            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residuals Plot')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: Metrics comparison bar chart
            ax3 = axes[1, 0]
            metrics = ['MAE', 'RMSE', 'R²']
            baseline_values = [baseline_mae, baseline_rmse, baseline_r2]
            enhanced_values = [enhanced_mae, enhanced_rmse, enhanced_r2]

            x = np.arange(len(metrics))
            width = 0.35

            bars1 = ax3.bar(x - width/2, baseline_values, width, label='Baseline', color='blue', alpha=0.8)
            bars2 = ax3.bar(x + width/2, enhanced_values, width, label='Enhanced', color='red', alpha=0.8)

            ax3.set_xlabel('Metrics')
            ax3.set_ylabel('Score')
            ax3.set_title('Model Performance Metrics')
            ax3.set_xticks(x)
            ax3.set_xticklabels(metrics)
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            def autolabel(bars):
                for bar in bars:
                    height = bar.get_height()
                    ax3.annotate(f'{height:.3f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)

            autolabel(bars1)
            autolabel(bars2)

            # Plot 4: Feature importance comparison (if available)
            ax4 = axes[1, 1]

            # Get feature importance from both models
            baseline_importance = self.baseline_predictor.get_feature_importance()
            enhanced_importance = self.enhanced_predictor.get_feature_importance()

            if baseline_importance and enhanced_importance:
                # Remove intercept for clarity
                baseline_feat_imp = {k: v for k, v in baseline_importance.items() if k != 'intercept'}
                enhanced_feat_imp = {k: v for k, v in enhanced_importance.items() if k != 'intercept'}

                # Get top 10 features from baseline model
                top_features = sorted(baseline_feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                feature_names_top = [f[0] for f in top_features]
                baseline_vals = [baseline_feat_imp[f] for f in feature_names_top]
                enhanced_vals = [enhanced_feat_imp.get(f, 0) for f in feature_names_top]

                x_pos = np.arange(len(feature_names_top))
                width = 0.35

                bars1 = ax4.barh(x_pos - width/2, baseline_vals, width, label='Baseline', color='blue', alpha=0.8)
                bars2 = ax4.barh(x_pos + width/2, enhanced_vals, width, label='Enhanced', color='red', alpha=0.8)

                ax4.set_yticks(x_pos)
                ax4.set_yticklabels(feature_names_top)
                ax4.set_xlabel('Coefficient Value')
                ax4.set_title('Top 10 Feature Importance Comparison')
                ax4.legend()
                ax4.grid(True, alpha=0.3, axis='x')
                ax4.axvline(x=0, color='k', linestyle='-', alpha=0.5)
            else:
                ax4.text(0.5, 0.5, 'Feature importance\nnot available',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax4.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax4.set_title('Feature Importance Comparison')

            plt.tight_layout()

            # Save the plot
            plot_path = os.path.join(self.viz_dir, 'model_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to: {plot_path}")

            return True

        except Exception as e:
            print(f"Error creating model comparison visualization: {e}")
            import traceback
            traceback.print_exc()
            return False

    def plot_anomaly_detection_insights(self) -> bool:
        """
        Create comprehensive anomaly detection visualizations

        Returns:
            Success status
        """
        print("Creating anomaly detection insights visualization...")

        try:
            # Load data
            X, y, feature_names = self.load_and_prepare_data()

            # Load or train anomaly detector
            if self.anomaly_detector is None:
                print("Training anomaly detector for visualization...")
                self.anomaly_detector = IsolationForestDetector(contamination=0.1)
                self.anomaly_detector.train(X, feature_names=feature_names)

            # Get anomaly scores and predictions
            anomaly_scores = self.anomaly_detector.predict_anomaly_score(X)
            anomaly_predictions = self.anomaly_detector.detect_anomaly(X, threshold=0.7)

            # Get explanation for anomalies
            explanation = self.anomaly_detector.explain_anomaly(X, sample_idx=None, top_n=5)

            # Create time indices (assuming sequential data)
            time_indices = np.arange(len(X))

            # Create the visualization
            fig, axes = plt.subplots(3, 2, figsize=(18, 15))
            fig.suptitle('Anomaly Detection Insights', fontsize=16, fontweight='bold')

            # Plot 1: Anomaly scores over time
            ax1 = axes[0, 0]
            ax1.plot(time_indices, anomaly_scores, alpha=0.7, linewidth=1, color='purple')
            ax1.axhline(y=0.7, color='r', linestyle='--', alpha=0.8, label='Anomaly Threshold (0.7)')
            ax1.fill_between(time_indices, 0.7, 1, where=(anomaly_scores >= 0.7),
                           alpha=0.3, color='red', label='Anomaly Regions')

            ax1.set_xlabel('Time Index')
            ax1.set_ylabel('Anomaly Score')
            ax1.set_title('Anomaly Scores Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Anomaly score distribution
            ax2 = axes[0, 1]
            ax2.hist(anomaly_scores, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
            ax2.axvline(x=0.7, color='r', linestyle='--', alpha=0.8, label='Threshold')
            ax2.axvline(x=np.mean(anomaly_scores), color='g', linestyle='-', alpha=0.8, label=f'Mean ({np.mean(anomaly_scores):.3f})')

            ax2.set_xlabel('Anomaly Score')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Anomaly Scores')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')

            # Plot 3: Top features contributing to anomalies
            ax3 = axes[1, 0]
            if explanation and 'feature_importance' in explanation:
                feat_imp = explanation['feature_importance']
                if feat_imp:
                    # Get top 10 features
                    top_feat = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
                    feat_names, feat_vals = zip(*top_feat)

                    y_pos = np.arange(len(feat_names))
                    ax3.barh(y_pos, feat_vals, color='orange', alpha=0.8)
                    ax3.set_yticks(y_pos)
                    ax3.set_yticklabels(feat_names)
                    ax3.set_xlabel('Feature Importance')
                    ax3.set_title('Top 10 Features for Anomaly Detection')
                    ax3.grid(True, alpha=0.3, axis='x')
                else:
                    ax3.text(0.5, 0.5, 'Feature importance\nnot available',
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax3.transAxes, fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    ax3.set_title('Feature Importance for Anomaly Detection')
            else:
                ax3.text(0.5, 0.5, 'Feature importance\nnot available',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax3.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax3.set_title('Feature Importance for Anomaly Detection')

            # Plot 4: Anomaly detection results overlay on CPU usage (if we had actual CPU data)
            ax4 = axes[1, 1]
            # Since we don't have the actual CPU usage in the features directly in this context,
            # we'll show the relationship between anomaly scores and a key feature
            if len(feature_names) > 0 and 'cpu_usage_percent' in ''.join(feature_names).lower():
                # Find CPU-related feature
                cpu_features = [f for f in feature_names if 'cpu' in f.lower()]
                if cpu_features:
                    cpu_feat_idx = feature_names.index(cpu_features[0])
                    cpu_feature_values = X[:, cpu_feat_idx]

                    scatter = ax4.scatter(time_indices, cpu_feature_values,
                                        c=anomaly_scores, cmap='hot', alpha=0.7, s=20)
                    plt.colorbar(scatter, ax=ax4, label='Anomaly Score')
                    ax4.set_xlabel('Time Index')
                    ax4.set_ylabel(f'{cpu_features[0]} Value')
                    ax4.set_title('CPU Feature vs Time (colored by anomaly score)')
                    ax4.grid(True, alpha=0.3)
                else:
                    ax4.text(0.5, 0.5, 'CPU feature not\nfound in dataset',
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax4.transAxes, fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    ax4.set_title('CPU Feature Analysis')
            else:
                # Show first two features relationship
                if len(feature_names) >= 2:
                    scatter = ax4.scatter(X[:, 0], X[:, 1],
                                        c=anomaly_scores, cmap='hot', alpha=0.7, s=20)
                    plt.colorbar(scatter, ax=ax4, label='Anomaly Score')
                    ax4.set_xlabel(feature_names[0])
                    ax4.set_ylabel(feature_names[1])
                    ax4.set_title(f'{feature_names[0]} vs {feature_names[1]} (colored by anomaly score)')
                    ax4.grid(True, alpha=0.3)
                else:
                    ax4.text(0.5, 0.5, 'Insufficient features\nfor scatter plot',
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax4.transAxes, fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    ax4.set_title('Feature Relationship Analysis')

            # Plot 5: Anomaly statistics summary
            ax5 = axes[2, 0]
            n_total = len(X)
            n_anomalies = np.sum(anomaly_predictions)
            anomaly_percentage = (n_anomalies / n_total) * 100

            # Create a text summary
            stats_text = f"""
Anomaly Detection Statistics:
• Total Samples: {n_total:,}
• Detected Anomalies: {n_anomalies:,}
• Anomaly Percentage: {anomaly_percentage:.2f}%
• Score Range: [{np.min(anomaly_scores):.3f}, {np.max(anomaly_scores):.3f}]
• Mean Score: {np.mean(anomaly_scores):.3f}
• Std Score: {np.std(anomaly_scores):.3f}
            """

            ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            ax5.set_xlim(0, 1)
            ax5.set_ylim(0, 1)
            ax5.axis('off')
            ax5.set_title('Anomaly Detection Summary')

            # Plot 6: Feature correlation with anomaly scores
            ax6 = axes[2, 1]
            if len(feature_names) > 0 and X.shape[1] > 0:
                # Calculate correlation between each feature and anomaly scores
                correlations = []
                valid_features = []

                for i, feat_name in enumerate(feature_names):
                    if i < X.shape[1]:  # Make sure we don't go out of bounds
                        try:
                            corr = np.corrcoef(X[:, i], anomaly_scores)[0, 1]
                            if not np.isnan(corr):
                                correlations.append(abs(corr))  # Use absolute value for importance
                                valid_features.append(feat_name)
                        except:
                            pass

                if correlations and valid_features:
                    # Get top 10 features by correlation with anomaly scores
                    feat_corr_pairs = list(zip(valid_features, correlations))
                    feat_corr_pairs.sort(key=lambda x: x[1], reverse=True)
                    top_feat_corr = feat_corr_pairs[:10]

                    feat_names_top, corr_vals_top = zip(*top_feat_corr)

                    y_pos = np.arange(len(feat_names_top))
                    ax6.barh(y_pos, corr_vals_top, color='green', alpha=0.8)
                    ax6.set_yticks(y_pos)
                    ax6.set_yticklabels(feat_names_top)
                    ax6.set_xlabel('|Correlation| with Anomaly Score')
                    ax6.set_title('Top 10 Features Correlated with Anomaly Scores')
                    ax6.grid(True, alpha=0.3, axis='x')
                else:
                    ax6.text(0.5, 0.5, 'Could not compute\nfeature correlations',
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax6.transAxes, fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    ax6.set_title('Feature Correlation Analysis')
            else:
                ax6.text(0.5, 0.5, 'No features available\nfor correlation analysis',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax6.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax6.set_title('Feature Correlation Analysis')

            plt.tight_layout()

            # Save the plot
            plot_path = os.path.join(self.viz_dir, 'anomaly_detection_insights.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Anomaly detection insights plot saved to: {plot_path}")

            return True

        except Exception as e:
            print(f"Error creating anomaly detection visualization: {e}")
            import traceback
            traceback.print_exc()
            return False

    def plot_training_progress_dashboard(self) -> bool:
        """
        Create a training progress dashboard showing model learning curves and performance over time

        Returns:
            Success status
        """
        print("Creating training progress dashboard...")

        try:
            # Load data
            X, y, feature_names = self.load_and_prepare_data()

            # Create learning curves for different training set sizes
            from sklearn.model_selection import learning_curve
            from sklearn.linear_model import LinearRegression

            # Create the visualization
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Model Training Progress Dashboard', fontsize=16, fontweight='bold')

            # Plot 1: Learning curve for baseline model
            ax1 = axes[0, 0]
            train_sizes, train_scores, test_scores = learning_curve(
                LinearRegression(), X, y, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='neg_mean_absolute_error'
            )

            train_scores_mean = -np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = -np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            ax1.grid(True, alpha=0.3)
            ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1, color="r")
            ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1, color="g")
            ax1.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            ax1.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
            ax1.set_xlabel("Training Set Size")
            ax1.set_ylabel("MAE Score")
            ax1.set_title("Learning Curve (Baseline Model)")
            ax1.legend(loc="best")

            # Plot 2: Feature distribution analysis
            ax2 = axes[0, 1]
            if len(feature_names) > 0 and X.shape[1] > 0:
                # Show distribution of first few features
                n_features_to_show = min(5, len(feature_names), X.shape[1])
                for i in range(n_features_to_show):
                    ax2.hist(X[:, i], bins=30, alpha=0.6, label=feature_names[i])
                ax2.set_xlabel('Feature Value')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Feature Distributions')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No feature data\navailable',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax2.set_title('Feature Distributions')

            # Plot 3: Target variable distribution
            ax3 = axes[0, 2]
            ax3.hist(y, bins=50, alpha=0.7, edgecolor='black', color='lightcoral')
            ax3.axvline(x=np.mean(y), color='r', linestyle='--', alpha=0.8, label=f'Mean: {np.mean(y):.2f}')
            ax3.axvline(x=np.median(y), color='g', linestyle='-', alpha=0.8, label=f'Median: {np.median(y):.2f}')
            ax3.set_xlabel('CPU Usage (%)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Target Variable Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')

            # Plot 4: Feature correlation matrix
            ax4 = axes[1, 0]
            if len(feature_names) > 0 and X.shape[1] > 1:
                # Limit to reasonable size for visualization
                max_features_for_corr = min(15, X.shape[1])
                if X.shape[1] > max_features_for_corr:
                    # Select features with highest variance
                    variances = np.var(X, axis=0)
                    top_feature_indices = np.argsort(variances)[-max_features_for_corr:]
                    X_corr = X[:, top_feature_indices]
                    feat_names_corr = [feature_names[i] for i in top_feature_indices]
                else:
                    X_corr = X
                    feat_names_corr = feature_names

                # Calculate correlation matrix
                corr_matrix = np.corrcoef(X_corr.T)

                # Create heatmap
                im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                ax4.set_xticks(range(len(feat_names_corr)))
                ax4.set_yticks(range(len(feat_names_corr)))
                ax4.set_xticklabels(feat_names_corr, rotation=45, ha='right')
                ax4.set_yticklabels(feat_names_corr)
                plt.colorbar(im, ax=ax4)
                ax4.set_title('Feature Correlation Matrix')
            else:
                ax4.text(0.5, 0.5, 'Insufficient features\nfor correlation matrix',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax4.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax4.set_title('Feature Correlation Matrix')

            # Plot 5: Prediction error analysis
            ax5 = axes[1, 1]
            # Train a model and analyze prediction errors
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            residuals = y - y_pred

            ax5.scatter(y_pred, residuals, alpha=0.6, s=20)
            ax5.axhline(y=0, color='r', linestyle='--', alpha=0.8)
            ax5.set_xlabel('Predicted Values')
            ax5.set_ylabel('Residuals')
            ax5.set_title('Residuals vs Predicted Values')
            ax5.grid(True, alpha=0.3)

            # Add statistics text
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(np.mean(residuals**2))
            stats_text = f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}'
            ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # Plot 6: Model complexity vs performance
            ax6 = axes[1, 2]
            # Show how different numbers of features affect performance
            from sklearn.feature_selection import SelectKBest, f_regression

            if X.shape[1] > 1:
                k_values = range(1, min(X.shape[1], 11))
                train_scores = []
                test_scores = []

                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                for k in k_values:
                    selector = SelectKBest(f_regression, k=k)
                    X_train_k = selector.fit_transform(X_train, y_train)
                    X_test_k = selector.transform(X_test)

                    model_k = LinearRegression()
                    model_k.fit(X_train_k, y_train)

                    train_pred = model_k.predict(X_train_k)
                    test_pred = model_k.predict(X_test_k)

                    train_mae = np.mean(np.abs(y_train - train_pred))
                    test_mae = np.mean(np.abs(y_test - test_pred))

                    train_scores.append(train_mae)
                    test_scores.append(test_mae)

                ax6.plot(k_values, train_scores, 'o-', label='Training MAE', color='blue')
                ax6.plot(k_values, test_scores, 'o-', label='Test MAE', color='red')
                ax6.set_xlabel('Number of Features (k)')
                ax6.set_ylabel('MAE Score')
                ax6.set_title('Feature Selection Performance')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
            else:
                ax6.text(0.5, 0.5, 'Insufficient features\nfor selection analysis',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax6.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax6.set_title('Feature Selection Performance')

            plt.tight_layout()

            # Save the plot
            plot_path = os.path.join(self.viz_dir, 'training_progress_dashboard.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training progress dashboard saved to: {plot_path}")

            return True

        except Exception as e:
            print(f"Error creating training progress dashboard: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_comprehensive_report(self) -> bool:
        """
        Create a comprehensive HTML report with all visualizations

        Returns:
            Success status
        """
        print("Creating comprehensive HTML report...")

        try:
            report_path = os.path.join(self.viz_dir, 'opsnexus_ml_report.html')

            html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpsNexus-ML Comprehensive Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        h1 {
            text-align: center;
            color: #3498db;
            border-bottom: 3px solid #ecf0f1;
            padding-bottom: 20px;
        }
        h2 {
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
        }
        .visualization {
            margin: 30px 0;
            text-align: center;
        }
        .visualization img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .stats-box {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .metric {
            display: inline-block;
            margin: 10px 15px 10px 0;
            padding: 10px 15px;
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            font-weight: bold;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #bdc3c7;
            color: #7f8c8d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>OpsNexus-ML Comprehensive Analysis Report</h1>
        <p><em>Generated on """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</em></p>

        <div class="stats-box">
            <h2>Executive Summary</h2>
            <p>This report provides a comprehensive analysis of the OpsNexus-ML machine learning intelligence layer, including model performance comparisons, anomaly detection insights, and training progress diagnostics.</p>
        </div>

        <h2>1. Model Performance Comparison</h2>
        <div class="visualization">
            <img src="model_comparison.png" alt="Model Performance Comparison">
        </div>
        <p>This visualization compares the performance of different machine learning algorithms (Linear Regression, Ridge, Lasso, Random Forest) for CPU usage prediction. It shows prediction accuracy, residuals comparison, performance metrics, and feature importance analysis.</p>

        <h2>2. Anomaly Detection Insights</h2>
        <div class="visualization">
            <img src="anomaly_detection_insights.png" alt="Anomaly Detection Insights">
        </div>
        <p>This visualization provides deep insights into the anomaly detection system, including anomaly score distributions, temporal patterns, feature contributions, and statistical summaries of detected anomalies.</p>

        <h2>3. Training Progress Dashboard</h2>
        <div class="visualization">
            <img src="training_progress_dashboard.png" alt="Training Progress Dashboard">
        </div>
        <p>This dashboard shows the model learning process, including learning curves, feature distributions, target variable analysis, correlation matrices, and feature selection performance.</p>

        <div class="footer">
            <p>OpsNexus-ML: Transforming system monitoring from observation to intelligent anticipation.</p>
            <p>Report generated by the OpsNexus-ML Visualization Suite</p>
        </div>
    </div>
</body>
</html>
            """

            with open(report_path, 'w') as f:
                f.write(html_content)

            print(f"Comprehensive HTML report saved to: {report_path}")
            return True

        except Exception as e:
            print(f"Error creating comprehensive report: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_all_visualizations(self) -> bool:
        """
        Run all visualization functions and create the comprehensive report

        Returns:
            Overall success status
        """
        print("OpsNexus-ML Enhanced Visualization Suite")
        print("=" * 50)

        # Load data and models
        try:
            self.load_and_prepare_data()
            self.load_models()
            print("Data and models loaded successfully!\n")
        except Exception as e:
            print(f"Error loading data/models: {e}")
            return False

        # Run all visualizations
        results = []

        print("1. Creating model comparison visualization...")
        results.append(self.plot_model_comparison())
        print()

        print("2. Creating anomaly detection insights visualization...")
        results.append(self.plot_anomaly_detection_insights())
        print()

        print("3. Creating training progress dashboard...")
        results.append(self.plot_training_progress_dashboard())
        print()

        print("4. Creating comprehensive HTML report...")
        results.append(self.create_comprehensive_report())
        print()

        # Summary
        success_count = sum(results)
        total_count = len(results)

        print(f"Visualization Suite Complete: {success_count}/{total_count} components successful")

        if success_count == total_count:
            print("✅ All visualizations created successfully!")
            print(f"📁 Output directory: {self.viz_dir}")
            return True
        else:
            print("❌ Some visualizations failed to create")
            return False


def main():
    """Main function to run the enhanced visualization suite"""
    visualizer = OpsNexusMLVisualizer()
    success = visualizer.run_all_visualizations()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()