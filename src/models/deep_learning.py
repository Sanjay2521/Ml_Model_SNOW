"""
Deep Learning Models
Implements DNN, RNN, and LSTM for text classification
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle


class DeepLearningModels:
    """Train and evaluate deep learning models"""

    def __init__(self, config: dict = None):
        """
        Initialize DeepLearningModels

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.models = {}
        self.tokenizer = None
        self.label_encoder = None
        self.max_sequence_length = 200
        self.vocab_size = 10000
        self.embedding_dim = 100

    def prepare_text_data(
        self,
        texts: List[str],
        labels: np.ndarray,
        max_words: int = 10000,
        max_len: int = 200,
        fit_tokenizer: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare text data for deep learning

        Args:
            texts: List of text documents
            labels: Target labels
            max_words: Maximum vocabulary size
            max_len: Maximum sequence length
            fit_tokenizer: Whether to fit tokenizer (True for training data)

        Returns:
            Tuple of (sequences, encoded_labels)
        """
        self.max_sequence_length = max_len
        self.vocab_size = max_words

        # Tokenize texts
        if fit_tokenizer:
            self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
            self.tokenizer.fit_on_texts(texts)

        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(
            sequences,
            maxlen=max_len,
            padding='post',
            truncating='post'
        )

        # Encode labels
        if fit_tokenizer:
            self.label_encoder = LabelEncoder()
            encoded_labels = self.label_encoder.fit_transform(labels)
        else:
            encoded_labels = self.label_encoder.transform(labels)

        return padded_sequences, encoded_labels

    def build_dnn(
        self,
        input_dim: int,
        num_classes: int,
        hidden_layers: List[int] = [256, 128, 64],
        dropout: float = 0.3,
        activation: str = 'relu'
    ) -> keras.Model:
        """
        Build Deep Neural Network

        Args:
            input_dim: Input dimension
            num_classes: Number of output classes
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate
            activation: Activation function

        Returns:
            Keras model
        """
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(hidden_layers[0], activation=activation),
            layers.Dropout(dropout),
            layers.BatchNormalization()
        ])

        # Add remaining hidden layers
        for units in hidden_layers[1:]:
            model.add(layers.Dense(units, activation=activation))
            model.add(layers.Dropout(dropout))
            model.add(layers.BatchNormalization())

        # Output layer
        if num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(num_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'

        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['accuracy']
        )

        return model

    def build_lstm(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_length: int,
        num_classes: int,
        lstm_units: List[int] = [128, 64],
        dropout: float = 0.2,
        recurrent_dropout: float = 0.2
    ) -> keras.Model:
        """
        Build LSTM model

        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            max_length: Maximum sequence length
            num_classes: Number of output classes
            lstm_units: List of LSTM layer sizes
            dropout: Dropout rate
            recurrent_dropout: Recurrent dropout rate

        Returns:
            Keras model
        """
        model = models.Sequential([
            layers.Input(shape=(max_length,)),
            layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            layers.SpatialDropout1D(dropout)
        ])

        # Add LSTM layers
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            model.add(layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout
            ))

        # Output layer
        if num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(num_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'

        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['accuracy']
        )

        return model

    def build_rnn(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_length: int,
        num_classes: int,
        rnn_units: List[int] = [128, 64],
        dropout: float = 0.2
    ) -> keras.Model:
        """
        Build RNN model

        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            max_length: Maximum sequence length
            num_classes: Number of output classes
            rnn_units: List of RNN layer sizes
            dropout: Dropout rate

        Returns:
            Keras model
        """
        model = models.Sequential([
            layers.Input(shape=(max_length,)),
            layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            layers.SpatialDropout1D(dropout)
        ])

        # Add SimpleRNN layers
        for i, units in enumerate(rnn_units):
            return_sequences = i < len(rnn_units) - 1
            model.add(layers.SimpleRNN(
                units,
                return_sequences=return_sequences,
                dropout=dropout
            ))

        # Output layer
        if num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(num_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'

        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['accuracy']
        )

        return model

    def build_bidirectional_lstm(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_length: int,
        num_classes: int,
        lstm_units: List[int] = [128, 64],
        dropout: float = 0.2
    ) -> keras.Model:
        """
        Build Bidirectional LSTM model

        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            max_length: Maximum sequence length
            num_classes: Number of output classes
            lstm_units: List of LSTM layer sizes
            dropout: Dropout rate

        Returns:
            Keras model
        """
        model = models.Sequential([
            layers.Input(shape=(max_length,)),
            layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            layers.SpatialDropout1D(dropout)
        ])

        # Add Bidirectional LSTM layers
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            model.add(layers.Bidirectional(
                layers.LSTM(units, return_sequences=return_sequences, dropout=dropout)
            ))

        # Output layer
        if num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(num_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'

        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['accuracy']
        )

        return model

    def train_model(
        self,
        model_name: str,
        model: keras.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 50,
        batch_size: int = 32,
        callbacks_list: List = None
    ) -> keras.callbacks.History:
        """
        Train deep learning model

        Args:
            model_name: Name of the model
            model: Keras model
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            callbacks_list: List of callbacks

        Returns:
            Training history
        """
        print(f"\nTraining {model_name}...")
        print(f"Model architecture:")
        model.summary()

        # Default callbacks
        if callbacks_list is None:
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7
                )
            ]

        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        # Train model
        history = model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )

        self.models[model_name] = model

        print(f"{model_name} training completed")

        return history

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

        predictions = self.models[model_name].predict(X)

        # Convert probabilities to class labels
        if predictions.shape[1] == 1:  # Binary classification
            predictions = (predictions > 0.5).astype(int).flatten()
        else:  # Multi-class classification
            predictions = np.argmax(predictions, axis=1)

        return predictions

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

        return self.models[model_name].predict(X)

    def save_model(self, model_name: str, file_path: str):
        """
        Save model to file

        Args:
            model_name: Name of the model
            file_path: Output file path
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")

        self.models[model_name].save(file_path)
        print(f"Model {model_name} saved to: {file_path}")

    def load_model(self, model_name: str, file_path: str):
        """
        Load model from file

        Args:
            model_name: Name to assign to the model
            file_path: Input file path
        """
        self.models[model_name] = keras.models.load_model(file_path)
        print(f"Model {model_name} loaded from: {file_path}")

    def save_tokenizer(self, file_path: str):
        """Save tokenizer to file"""
        with open(file_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"Tokenizer saved to: {file_path}")

    def load_tokenizer(self, file_path: str):
        """Load tokenizer from file"""
        with open(file_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        print(f"Tokenizer loaded from: {file_path}")

    def save_label_encoder(self, file_path: str):
        """Save label encoder to file"""
        with open(file_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"Label encoder saved to: {file_path}")

    def load_label_encoder(self, file_path: str):
        """Load label encoder from file"""
        with open(file_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print(f"Label encoder loaded from: {file_path}")


if __name__ == "__main__":
    # Example usage
    dl_models = DeepLearningModels()

    # Sample data
    texts = [
        "email application not working",
        "printer issue paper jam",
        "vpn connection problem",
        "password reset needed",
        "application crash error"
    ] * 100

    labels = np.array(['IT Support', 'Hardware', 'Network', 'IT Support', 'Application'] * 100)

    # Prepare data
    X, y = dl_models.prepare_text_data(texts, labels)

    print(f"Sequences shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    # Build LSTM model
    num_classes = len(np.unique(y))
    lstm_model = dl_models.build_lstm(
        vocab_size=dl_models.vocab_size,
        embedding_dim=100,
        max_length=dl_models.max_sequence_length,
        num_classes=num_classes
    )

    print("LSTM model built successfully!")
