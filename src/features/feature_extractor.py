"""
Feature Engineering Module
Implements tokenization, vectorization, and feature extraction
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Union
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.chunk import ne_chunk
from sklearn.preprocessing import LabelEncoder
import pickle

# Download required NLTK data
try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('words')
except LookupError:
    nltk.download('words')


class FeatureExtractor:
    """Extract features from text data"""

    def __init__(self, config: dict = None):
        """
        Initialize FeatureExtractor

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.label_encoders = {}

    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        if pd.isna(text) or text == "":
            return []
        return word_tokenize(str(text))

    def pos_tagging(self, text: str) -> List[Tuple[str, str]]:
        """
        Perform Part-of-Speech tagging

        Args:
            text: Input text

        Returns:
            List of (word, pos_tag) tuples
        """
        tokens = self.tokenize_text(text)
        return pos_tag(tokens)

    def extract_pos_features(self, text: str) -> dict:
        """
        Extract POS tag counts as features

        Args:
            text: Input text

        Returns:
            Dictionary of POS tag counts
        """
        pos_tags = self.pos_tagging(text)

        pos_counts = {
            'noun_count': 0,
            'verb_count': 0,
            'adj_count': 0,
            'adv_count': 0,
            'total_tokens': len(pos_tags)
        }

        for word, tag in pos_tags:
            if tag.startswith('NN'):  # Nouns
                pos_counts['noun_count'] += 1
            elif tag.startswith('VB'):  # Verbs
                pos_counts['verb_count'] += 1
            elif tag.startswith('JJ'):  # Adjectives
                pos_counts['adj_count'] += 1
            elif tag.startswith('RB'):  # Adverbs
                pos_counts['adv_count'] += 1

        return pos_counts

    def named_entity_recognition(self, text: str) -> List[str]:
        """
        Extract named entities from text

        Args:
            text: Input text

        Returns:
            List of named entities
        """
        tokens = self.tokenize_text(text)
        pos_tags = pos_tag(tokens)
        chunks = ne_chunk(pos_tags, binary=False)

        named_entities = []
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                entity = ' '.join(c[0] for c in chunk)
                named_entities.append(entity)

        return named_entities

    def extract_ner_features(self, text: str) -> dict:
        """
        Extract NER counts as features

        Args:
            text: Input text

        Returns:
            Dictionary of NER features
        """
        entities = self.named_entity_recognition(text)

        return {
            'entity_count': len(entities),
            'has_entities': int(len(entities) > 0)
        }

    def extract_text_statistics(self, text: str) -> dict:
        """
        Extract statistical features from text

        Args:
            text: Input text

        Returns:
            Dictionary of text statistics
        """
        if pd.isna(text) or text == "":
            return {
                'char_count': 0,
                'word_count': 0,
                'avg_word_length': 0,
                'sentence_count': 0,
                'uppercase_count': 0,
                'digit_count': 0
            }

        text_str = str(text)
        words = text_str.split()

        return {
            'char_count': len(text_str),
            'word_count': len(words),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'sentence_count': text_str.count('.') + text_str.count('!') + text_str.count('?'),
            'uppercase_count': sum(1 for c in text_str if c.isupper()),
            'digit_count': sum(1 for c in text_str if c.isdigit())
        }

    def encode_categorical_features(
        self,
        df: pd.DataFrame,
        categorical_columns: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding

        Args:
            df: Input dataframe
            categorical_columns: List of categorical column names
            fit: Whether to fit the encoder (True for training data)

        Returns:
            Dataframe with encoded features
        """
        df_encoded = df.copy()

        for col in categorical_columns:
            if col not in df_encoded.columns:
                continue

            # Fill missing values
            df_encoded[col] = df_encoded[col].fillna('Unknown')

            if fit:
                # Fit and transform
                le = LabelEncoder()
                df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le
            else:
                # Transform using existing encoder
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen labels
                    df_encoded[f'{col}_encoded'] = df_encoded[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                else:
                    raise ValueError(f"No fitted encoder found for column: {col}")

        return df_encoded

    def extract_all_features(
        self,
        df: pd.DataFrame,
        text_column: str = 'cleaned_text',
        categorical_columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Extract all features from dataframe

        Args:
            df: Input dataframe
            text_column: Column containing cleaned text
            categorical_columns: List of categorical columns to encode

        Returns:
            Dataframe with all extracted features
        """
        df_features = df.copy()

        if text_column in df_features.columns:
            print(f"Extracting features from '{text_column}'...")

            # Text statistics
            print("Extracting text statistics...")
            text_stats = df_features[text_column].apply(self.extract_text_statistics)
            text_stats_df = pd.DataFrame(text_stats.tolist())
            df_features = pd.concat([df_features, text_stats_df], axis=1)

            # POS features (can be slow for large datasets)
            if self.config.get('nlp_features', {}).get('pos_tagging', False):
                print("Extracting POS features...")
                pos_features = df_features[text_column].apply(self.extract_pos_features)
                pos_features_df = pd.DataFrame(pos_features.tolist())
                df_features = pd.concat([df_features, pos_features_df], axis=1)

            # NER features (can be slow for large datasets)
            if self.config.get('nlp_features', {}).get('ner', False):
                print("Extracting NER features...")
                ner_features = df_features[text_column].apply(self.extract_ner_features)
                ner_features_df = pd.DataFrame(ner_features.tolist())
                df_features = pd.concat([df_features, ner_features_df], axis=1)

        # Encode categorical features
        if categorical_columns:
            print("Encoding categorical features...")
            df_features = self.encode_categorical_features(df_features, categorical_columns)

        return df_features

    def save_encoders(self, file_path: str):
        """Save label encoders to file"""
        with open(file_path, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print(f"Label encoders saved to: {file_path}")

    def load_encoders(self, file_path: str):
        """Load label encoders from file"""
        with open(file_path, 'rb') as f:
            self.label_encoders = pickle.load(f)
        print(f"Label encoders loaded from: {file_path}")


class FeatureSelector:
    """Select best features for modeling"""

    def __init__(self, method: str = 'mutual_info'):
        """
        Initialize FeatureSelector

        Args:
            method: Feature selection method ('mutual_info', 'chi2', 'variance')
        """
        self.method = method
        self.selected_features = None

    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Select top k features

        Args:
            X: Feature matrix
            y: Target vector
            k: Number of features to select

        Returns:
            Tuple of (selected features, feature indices)
        """
        from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2

        if self.method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=k)
        elif self.method == 'chi2':
            selector = SelectKBest(chi2, k=k)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

        X_selected = selector.fit_transform(X, y)
        self.selected_features = selector.get_support(indices=True)

        print(f"Selected {k} features using {self.method}")
        return X_selected, self.selected_features


if __name__ == "__main__":
    # Example usage
    extractor = FeatureExtractor()

    sample_text = "The email application is not working properly"

    # Tokenization
    tokens = extractor.tokenize_text(sample_text)
    print(f"Tokens: {tokens}")

    # POS Tagging
    pos_tags = extractor.pos_tagging(sample_text)
    print(f"POS Tags: {pos_tags}")

    # POS Features
    pos_features = extractor.extract_pos_features(sample_text)
    print(f"POS Features: {pos_features}")

    # Text Statistics
    text_stats = extractor.extract_text_statistics(sample_text)
    print(f"Text Statistics: {text_stats}")
