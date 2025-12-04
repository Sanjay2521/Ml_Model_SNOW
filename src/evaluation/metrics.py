"""
Evaluation Metrics Module
Implements Accuracy, Precision, Recall, F1 Score, Cohen Kappa Score, Confusion Matrix
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from pathlib import Path
import json


class ModelEvaluator:
    """Evaluate model performance with various metrics"""

    def __init__(self, config: dict = None):
        """
        Initialize ModelEvaluator

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.results = {}

    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None,
        model_name: str = 'model',
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            model_name: Name of the model
            average: Averaging method for multi-class metrics

        Returns:
            Dictionary of metrics
        """
        print(f"\nCalculating metrics for {model_name}...")

        metrics = {}

        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # Precision
        metrics['precision'] = precision_score(
            y_true, y_pred, average=average, zero_division=0
        )

        # Recall
        metrics['recall'] = recall_score(
            y_true, y_pred, average=average, zero_division=0
        )

        # F1 Score
        metrics['f1_score'] = f1_score(
            y_true, y_pred, average=average, zero_division=0
        )

        # Cohen Kappa Score
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

        # ROC AUC (if probabilities available)
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:  # Multi-class
                    metrics['roc_auc'] = roc_auc_score(
                        y_true, y_pred_proba, multi_class='ovr', average=average
                    )
            except Exception as e:
                print(f"Could not calculate ROC AUC: {e}")
                metrics['roc_auc'] = None

        self.results[model_name] = metrics

        return metrics

    def print_metrics(self, model_name: str):
        """
        Print metrics for a model

        Args:
            model_name: Name of the model
        """
        if model_name not in self.results:
            raise ValueError(f"No results found for {model_name}")

        print(f"\n{'='*50}")
        print(f"Evaluation Metrics for {model_name}")
        print(f"{'='*50}")

        for metric, value in self.results[model_name].items():
            if value is not None:
                print(f"{metric:20s}: {value:.4f}")
            else:
                print(f"{metric:20s}: N/A")

        print(f"{'='*50}\n")

    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Calculate confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str] = None,
        title: str = 'Confusion Matrix',
        save_path: str = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            title: Plot title
            save_path: Path to save the plot
            figsize: Figure size
        """
        cm = self.get_confusion_matrix(y_true, y_pred)

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar=True
        )
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")

        plt.show()

    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str] = None,
        output_dict: bool = False
    ):
        """
        Get classification report

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            output_dict: Return as dictionary

        Returns:
            Classification report (string or dict)
        """
        return classification_report(
            y_true,
            y_pred,
            target_names=labels,
            output_dict=output_dict,
            zero_division=0
        )

    def print_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str] = None,
        model_name: str = 'Model'
    ):
        """
        Print classification report

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            model_name: Name of the model
        """
        print(f"\n{'='*60}")
        print(f"Classification Report for {model_name}")
        print(f"{'='*60}")
        print(self.get_classification_report(y_true, y_pred, labels))
        print(f"{'='*60}\n")

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = 'Model',
        save_path: str = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Plot ROC curve (for binary classification)

        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            model_name: Name of the model
            save_path: Path to save the plot
            figsize: Figure size
        """
        if len(np.unique(y_true)) != 2:
            print("ROC curve is only for binary classification")
            return

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        auc = roc_auc_score(y_true, y_pred_proba[:, 1])

        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to: {save_path}")

        plt.show()

    def compare_models(
        self,
        save_path: str = None
    ) -> pd.DataFrame:
        """
        Compare all evaluated models

        Args:
            save_path: Path to save comparison table

        Returns:
            DataFrame with comparison results
        """
        if not self.results:
            raise ValueError("No models evaluated yet")

        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.round(4)

        # Sort by F1 score
        if 'f1_score' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('f1_score', ascending=False)

        print("\n" + "="*80)
        print("Model Comparison")
        print("="*80)
        print(comparison_df.to_string())
        print("="*80 + "\n")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            comparison_df.to_csv(save_path)
            print(f"Comparison saved to: {save_path}")

        return comparison_df

    def plot_model_comparison(
        self,
        metrics: List[str] = None,
        save_path: str = None,
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        Plot model comparison chart

        Args:
            metrics: List of metrics to compare
            save_path: Path to save the plot
            figsize: Figure size
        """
        if not self.results:
            raise ValueError("No models evaluated yet")

        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'cohen_kappa']

        # Create comparison dataframe
        comparison_df = pd.DataFrame(self.results).T

        # Filter available metrics
        available_metrics = [m for m in metrics if m in comparison_df.columns]

        if not available_metrics:
            print("No metrics available for comparison")
            return

        # Create subplots
        fig, axes = plt.subplots(1, len(available_metrics), figsize=figsize)

        if len(available_metrics) == 1:
            axes = [axes]

        for idx, metric in enumerate(available_metrics):
            data = comparison_df[metric].dropna().sort_values(ascending=False)

            axes[idx].barh(data.index, data.values, color='skyblue', edgecolor='navy')
            axes[idx].set_xlabel('Score', fontsize=10)
            axes[idx].set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            axes[idx].grid(axis='x', alpha=0.3)
            axes[idx].set_xlim([0, 1])

            # Add value labels
            for i, v in enumerate(data.values):
                axes[idx].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison chart saved to: {save_path}")

        plt.show()

    def save_results(self, file_path: str):
        """
        Save evaluation results to JSON

        Args:
            file_path: Output file path
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as f:
            json.dump(self.results, f, indent=4)

        print(f"Results saved to: {file_path}")

    def load_results(self, file_path: str):
        """
        Load evaluation results from JSON

        Args:
            file_path: Input file path
        """
        with open(file_path, 'r') as f:
            self.results = json.load(f)

        print(f"Results loaded from: {file_path}")


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_all_metrics(y_test, y_pred, y_pred_proba, 'Random Forest')
    evaluator.print_metrics('Random Forest')
    evaluator.print_classification_report(y_test, y_pred, model_name='Random Forest')
