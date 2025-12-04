"""
Traditional Machine Learning Models
Implements KNN, SVM, Decision Tree, Random Forest, Naive Bayes, Logistic Regression, Gradient Boosting, SGD
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import pickle
import joblib


class TraditionalMLModels:
    """Train and evaluate traditional ML models"""

    def __init__(self, config: dict = None):
        """
        Initialize TraditionalMLModels

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}

    def get_logistic_regression(self, params: dict = None) -> LogisticRegression:
        """
        Get Logistic Regression model

        Args:
            params: Model parameters

        Returns:
            LogisticRegression instance
        """
        default_params = {
            'max_iter': 1000,
            'solver': 'lbfgs',
            'multi_class': 'auto',
            'random_state': 42
        }

        if params:
            default_params.update(params)

        return LogisticRegression(**default_params)

    def get_random_forest(self, params: dict = None) -> RandomForestClassifier:
        """
        Get Random Forest model

        Args:
            params: Model parameters

        Returns:
            RandomForestClassifier instance
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'random_state': 42,
            'n_jobs': -1
        }

        if params:
            default_params.update(params)

        return RandomForestClassifier(**default_params)

    def get_svm(self, params: dict = None) -> SVC:
        """
        Get SVM model

        Args:
            params: Model parameters

        Returns:
            SVC instance
        """
        default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'probability': True,
            'random_state': 42
        }

        if params:
            default_params.update(params)

        return SVC(**default_params)

    def get_naive_bayes(self, model_type: str = 'multinomial', params: dict = None):
        """
        Get Naive Bayes model

        Args:
            model_type: 'multinomial' or 'gaussian'
            params: Model parameters

        Returns:
            Naive Bayes instance
        """
        default_params = params or {}

        if model_type == 'multinomial':
            return MultinomialNB(**default_params)
        elif model_type == 'gaussian':
            return GaussianNB(**default_params)
        else:
            raise ValueError(f"Unsupported Naive Bayes type: {model_type}")

    def get_knn(self, params: dict = None) -> KNeighborsClassifier:
        """
        Get KNN model

        Args:
            params: Model parameters

        Returns:
            KNeighborsClassifier instance
        """
        default_params = {
            'n_neighbors': 5,
            'weights': 'uniform',
            'n_jobs': -1
        }

        if params:
            default_params.update(params)

        return KNeighborsClassifier(**default_params)

    def get_decision_tree(self, params: dict = None) -> DecisionTreeClassifier:
        """
        Get Decision Tree model

        Args:
            params: Model parameters

        Returns:
            DecisionTreeClassifier instance
        """
        default_params = {
            'max_depth': None,
            'min_samples_split': 2,
            'random_state': 42
        }

        if params:
            default_params.update(params)

        return DecisionTreeClassifier(**default_params)

    def get_gradient_boosting(self, params: dict = None) -> GradientBoostingClassifier:
        """
        Get Gradient Boosting model

        Args:
            params: Model parameters

        Returns:
            GradientBoostingClassifier instance
        """
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        }

        if params:
            default_params.update(params)

        return GradientBoostingClassifier(**default_params)

    def get_sgd(self, params: dict = None) -> SGDClassifier:
        """
        Get SGD Classifier model

        Args:
            params: Model parameters

        Returns:
            SGDClassifier instance
        """
        default_params = {
            'loss': 'log_loss',
            'max_iter': 1000,
            'random_state': 42,
            'n_jobs': -1
        }

        if params:
            default_params.update(params)

        return SGDClassifier(**default_params)

    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: dict = None
    ) -> Any:
        """
        Train a single model

        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            params: Model parameters

        Returns:
            Trained model
        """
        print(f"\nTraining {model_name}...")

        # Get model instance
        if model_name == 'logistic_regression':
            model = self.get_logistic_regression(params)
        elif model_name == 'random_forest':
            model = self.get_random_forest(params)
        elif model_name == 'svm':
            model = self.get_svm(params)
        elif model_name == 'naive_bayes':
            model = self.get_naive_bayes(params=params)
        elif model_name == 'knn':
            model = self.get_knn(params)
        elif model_name == 'decision_tree':
            model = self.get_decision_tree(params)
        elif model_name == 'gradient_boosting':
            model = self.get_gradient_boosting(params)
        elif model_name == 'sgd':
            model = self.get_sgd(params)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Train model
        model.fit(X_train, y_train)
        self.models[model_name] = model

        print(f"{model_name} training completed")

        return model

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        models_config: dict = None
    ) -> Dict[str, Any]:
        """
        Train all enabled models

        Args:
            X_train: Training features
            y_train: Training labels
            models_config: Configuration for models

        Returns:
            Dictionary of trained models
        """
        if models_config is None:
            models_config = self.config.get('models', {}).get('traditional', {})

        for model_name, model_config in models_config.items():
            if model_config.get('enabled', True):
                params = {k: v for k, v in model_config.items() if k != 'enabled'}
                self.train_model(model_name, X_train, y_train, params)

        return self.models

    def hyperparameter_tuning(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: dict,
        method: str = 'grid',
        cv: int = 5,
        scoring: str = 'f1_weighted'
    ) -> Tuple[Any, dict]:
        """
        Perform hyperparameter tuning

        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for search
            method: 'grid' or 'random'
            cv: Number of cross-validation folds
            scoring: Scoring metric

        Returns:
            Tuple of (best model, best parameters)
        """
        print(f"\nPerforming hyperparameter tuning for {model_name}...")

        # Get base model
        if model_name == 'logistic_regression':
            base_model = self.get_logistic_regression()
        elif model_name == 'random_forest':
            base_model = self.get_random_forest()
        elif model_name == 'svm':
            base_model = self.get_svm()
        elif model_name == 'knn':
            base_model = self.get_knn()
        elif model_name == 'decision_tree':
            base_model = self.get_decision_tree()
        elif model_name == 'gradient_boosting':
            base_model = self.get_gradient_boosting()
        elif model_name == 'sgd':
            base_model = self.get_sgd()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Perform search
        if method == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
        elif method == 'random':
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=20,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported method: {method}")

        search.fit(X_train, y_train)

        print(f"Best parameters: {search.best_params_}")
        print(f"Best score: {search.best_score_:.4f}")

        self.best_params[model_name] = search.best_params_
        self.models[model_name] = search.best_estimator_

        return search.best_estimator_, search.best_params_

    def cross_validate_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5,
        scoring: str = 'f1_weighted'
    ) -> np.ndarray:
        """
        Perform cross-validation

        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            cv: Number of folds
            scoring: Scoring metric

        Returns:
            Array of cross-validation scores
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")

        print(f"\nPerforming {cv}-fold cross-validation for {model_name}...")

        scores = cross_val_score(
            self.models[model_name],
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        self.cv_scores[model_name] = scores

        print(f"CV Scores: {scores}")
        print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        return scores

    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            model_name: Name of the model
            X: Features

        Returns:
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")

        return self.models[model_name].predict(X)

    def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions

        Args:
            model_name: Name of the model
            X: Features

        Returns:
            Prediction probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")

        if hasattr(self.models[model_name], 'predict_proba'):
            return self.models[model_name].predict_proba(X)
        else:
            raise AttributeError(f"Model {model_name} does not support probability predictions")

    def save_model(self, model_name: str, file_path: str, format: str = 'pickle'):
        """
        Save model to file

        Args:
            model_name: Name of the model
            file_path: Output file path
            format: 'pickle' or 'joblib'
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")

        if format == 'pickle':
            with open(file_path, 'wb') as f:
                pickle.dump(self.models[model_name], f)
        elif format == 'joblib':
            joblib.dump(self.models[model_name], file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Model {model_name} saved to: {file_path}")

    def load_model(self, model_name: str, file_path: str, format: str = 'pickle'):
        """
        Load model from file

        Args:
            model_name: Name to assign to the model
            file_path: Input file path
            format: 'pickle' or 'joblib'
        """
        if format == 'pickle':
            with open(file_path, 'rb') as f:
                self.models[model_name] = pickle.load(f)
        elif format == 'joblib':
            self.models[model_name] = joblib.load(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Model {model_name} loaded from: {file_path}")


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    ml_models = TraditionalMLModels()

    # Train a single model
    model = ml_models.train_model('random_forest', X_train, y_train)
    predictions = ml_models.predict('random_forest', X_test)

    print(f"Predictions shape: {predictions.shape}")
