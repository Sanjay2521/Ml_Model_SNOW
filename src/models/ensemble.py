"""
Ensemble Methods
Implements Voting Classifier and Stacking
"""

import numpy as np
from typing import List, Dict, Any
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import pickle


class EnsembleModels:
    """Create and train ensemble models"""

    def __init__(self, config: dict = None):
        """
        Initialize EnsembleModels

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.ensemble_models = {}

    def create_voting_classifier(
        self,
        estimators: List[tuple],
        voting: str = 'soft',
        weights: List[float] = None
    ) -> VotingClassifier:
        """
        Create Voting Classifier

        Args:
            estimators: List of (name, estimator) tuples
            voting: 'hard' or 'soft'
            weights: Voting weights for each estimator

        Returns:
            VotingClassifier instance
        """
        print(f"Creating Voting Classifier with {len(estimators)} models...")

        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights,
            n_jobs=-1
        )

        return ensemble

    def create_stacking_classifier(
        self,
        estimators: List[tuple],
        final_estimator: Any = None,
        cv: int = 5
    ) -> StackingClassifier:
        """
        Create Stacking Classifier

        Args:
            estimators: List of (name, estimator) tuples for base models
            final_estimator: Meta-learner (default: LogisticRegression)
            cv: Number of cross-validation folds

        Returns:
            StackingClassifier instance
        """
        print(f"Creating Stacking Classifier with {len(estimators)} base models...")

        if final_estimator is None:
            final_estimator = LogisticRegression(max_iter=1000)

        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            n_jobs=-1
        )

        return ensemble

    def train_voting_ensemble(
        self,
        name: str,
        estimators: List[tuple],
        X_train: np.ndarray,
        y_train: np.ndarray,
        voting: str = 'soft',
        weights: List[float] = None
    ) -> VotingClassifier:
        """
        Train Voting Classifier

        Args:
            name: Name for the ensemble
            estimators: List of (name, estimator) tuples
            X_train: Training features
            y_train: Training labels
            voting: 'hard' or 'soft'
            weights: Voting weights

        Returns:
            Trained VotingClassifier
        """
        print(f"\nTraining Voting Ensemble: {name}")

        ensemble = self.create_voting_classifier(estimators, voting, weights)
        ensemble.fit(X_train, y_train)

        self.ensemble_models[name] = ensemble
        print(f"Voting Ensemble {name} trained successfully")

        return ensemble

    def train_stacking_ensemble(
        self,
        name: str,
        estimators: List[tuple],
        X_train: np.ndarray,
        y_train: np.ndarray,
        final_estimator: Any = None,
        cv: int = 5
    ) -> StackingClassifier:
        """
        Train Stacking Classifier

        Args:
            name: Name for the ensemble
            estimators: List of (name, estimator) tuples
            X_train: Training features
            y_train: Training labels
            final_estimator: Meta-learner
            cv: Number of CV folds

        Returns:
            Trained StackingClassifier
        """
        print(f"\nTraining Stacking Ensemble: {name}")

        ensemble = self.create_stacking_classifier(estimators, final_estimator, cv)
        ensemble.fit(X_train, y_train)

        self.ensemble_models[name] = ensemble
        print(f"Stacking Ensemble {name} trained successfully")

        return ensemble

    def create_two_level_model(
        self,
        level1_models: List[tuple],
        level2_model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> StackingClassifier:
        """
        Create two-level modeling (Level 1 / Level 2)

        Args:
            level1_models: Base models for Level 1
            level2_model: Meta-model for Level 2
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained two-level model
        """
        print("Creating Two-Level Model...")
        print(f"Level 1: {len(level1_models)} base models")
        print(f"Level 2: Meta-learner")

        two_level_model = self.create_stacking_classifier(
            estimators=level1_models,
            final_estimator=level2_model,
            cv=5
        )

        two_level_model.fit(X_train, y_train)

        self.ensemble_models['two_level'] = two_level_model
        print("Two-Level Model trained successfully")

        return two_level_model

    def predict(self, name: str, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            name: Name of the ensemble
            X: Features

        Returns:
            Predictions
        """
        if name not in self.ensemble_models:
            raise ValueError(f"Ensemble {name} not trained yet")

        return self.ensemble_models[name].predict(X)

    def predict_proba(self, name: str, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions

        Args:
            name: Name of the ensemble
            X: Features

        Returns:
            Prediction probabilities
        """
        if name not in self.ensemble_models:
            raise ValueError(f"Ensemble {name} not trained yet")

        return self.ensemble_models[name].predict_proba(X)

    def save_ensemble(self, name: str, file_path: str):
        """
        Save ensemble model

        Args:
            name: Name of the ensemble
            file_path: Output file path
        """
        if name not in self.ensemble_models:
            raise ValueError(f"Ensemble {name} not trained yet")

        with open(file_path, 'wb') as f:
            pickle.dump(self.ensemble_models[name], f)

        print(f"Ensemble {name} saved to: {file_path}")

    def load_ensemble(self, name: str, file_path: str):
        """
        Load ensemble model

        Args:
            name: Name to assign to the ensemble
            file_path: Input file path
        """
        with open(file_path, 'rb') as f:
            self.ensemble_models[name] = pickle.load(f)

        print(f"Ensemble {name} loaded from: {file_path}")


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create base models
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('svm', SVC(probability=True, random_state=42))
    ]

    # Initialize ensemble
    ensemble = EnsembleModels()

    # Train voting ensemble
    voting_model = ensemble.train_voting_ensemble(
        'voting',
        estimators,
        X_train,
        y_train,
        voting='soft'
    )

    # Make predictions
    predictions = ensemble.predict('voting', X_test)
    print(f"Predictions shape: {predictions.shape}")
