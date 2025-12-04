"""
Vectorization Module
Implements Count Vectorizer, TF-IDF, and Word2Vec
"""

import pandas as pd
import numpy as np
from typing import List, Union, Tuple
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle


class TextVectorizer:
    """Vectorize text using various methods"""

    def __init__(self, config: dict = None):
        """
        Initialize TextVectorizer

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.count_vectorizer = None
        self.tfidf_vectorizer = None
        self.word2vec_model = None
        self.doc2vec_model = None

    def fit_count_vectorizer(
        self,
        texts: List[str],
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        **kwargs
    ) -> CountVectorizer:
        """
        Fit Count Vectorizer

        Args:
            texts: List of text documents
            max_features: Maximum number of features
            ngram_range: N-gram range
            **kwargs: Additional CountVectorizer parameters

        Returns:
            Fitted CountVectorizer
        """
        print("Fitting Count Vectorizer...")

        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            **kwargs
        )

        self.count_vectorizer.fit(texts)
        print(f"Count Vectorizer fitted with {len(self.count_vectorizer.vocabulary_)} features")

        return self.count_vectorizer

    def transform_count_vectorizer(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts using Count Vectorizer

        Args:
            texts: List of text documents

        Returns:
            Count vector matrix
        """
        if self.count_vectorizer is None:
            raise ValueError("Count Vectorizer not fitted yet")

        return self.count_vectorizer.transform(texts).toarray()

    def fit_tfidf_vectorizer(
        self,
        texts: List[str],
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        **kwargs
    ) -> TfidfVectorizer:
        """
        Fit TF-IDF Vectorizer

        Args:
            texts: List of text documents
            max_features: Maximum number of features
            ngram_range: N-gram range
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            **kwargs: Additional TfidfVectorizer parameters

        Returns:
            Fitted TfidfVectorizer
        """
        print("Fitting TF-IDF Vectorizer...")

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            **kwargs
        )

        self.tfidf_vectorizer.fit(texts)
        print(f"TF-IDF Vectorizer fitted with {len(self.tfidf_vectorizer.vocabulary_)} features")

        return self.tfidf_vectorizer

    def transform_tfidf_vectorizer(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts using TF-IDF Vectorizer

        Args:
            texts: List of text documents

        Returns:
            TF-IDF vector matrix
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF Vectorizer not fitted yet")

        return self.tfidf_vectorizer.transform(texts).toarray()

    def fit_word2vec(
        self,
        texts: List[str],
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        epochs: int = 10,
        **kwargs
    ) -> Word2Vec:
        """
        Train Word2Vec model

        Args:
            texts: List of text documents
            vector_size: Dimensionality of word vectors
            window: Context window size
            min_count: Minimum word frequency
            workers: Number of worker threads
            epochs: Number of training epochs
            **kwargs: Additional Word2Vec parameters

        Returns:
            Trained Word2Vec model
        """
        print("Training Word2Vec model...")

        # Tokenize texts
        tokenized_texts = [text.split() for text in texts if text]

        self.word2vec_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs,
            **kwargs
        )

        print(f"Word2Vec model trained with vocabulary size: {len(self.word2vec_model.wv)}")

        return self.word2vec_model

    def text_to_word2vec(self, text: str, method: str = 'mean') -> np.ndarray:
        """
        Convert text to Word2Vec representation

        Args:
            text: Input text
            method: Aggregation method ('mean', 'sum', 'max')

        Returns:
            Document vector
        """
        if self.word2vec_model is None:
            raise ValueError("Word2Vec model not trained yet")

        words = text.split()
        word_vectors = []

        for word in words:
            if word in self.word2vec_model.wv:
                word_vectors.append(self.word2vec_model.wv[word])

        if not word_vectors:
            # Return zero vector if no words found
            return np.zeros(self.word2vec_model.vector_size)

        word_vectors = np.array(word_vectors)

        if method == 'mean':
            return np.mean(word_vectors, axis=0)
        elif method == 'sum':
            return np.sum(word_vectors, axis=0)
        elif method == 'max':
            return np.max(word_vectors, axis=0)
        else:
            raise ValueError(f"Unsupported method: {method}")

    def transform_word2vec(self, texts: List[str], method: str = 'mean') -> np.ndarray:
        """
        Transform texts to Word2Vec vectors

        Args:
            texts: List of text documents
            method: Aggregation method

        Returns:
            Document vector matrix
        """
        vectors = [self.text_to_word2vec(text, method) for text in texts]
        return np.array(vectors)

    def fit_doc2vec(
        self,
        texts: List[str],
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        epochs: int = 10,
        **kwargs
    ) -> Doc2Vec:
        """
        Train Doc2Vec model

        Args:
            texts: List of text documents
            vector_size: Dimensionality of document vectors
            window: Context window size
            min_count: Minimum word frequency
            workers: Number of worker threads
            epochs: Number of training epochs
            **kwargs: Additional Doc2Vec parameters

        Returns:
            Trained Doc2Vec model
        """
        print("Training Doc2Vec model...")

        # Create tagged documents
        tagged_docs = [
            TaggedDocument(words=text.split(), tags=[i])
            for i, text in enumerate(texts) if text
        ]

        self.doc2vec_model = Doc2Vec(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs,
            **kwargs
        )

        self.doc2vec_model.build_vocab(tagged_docs)
        self.doc2vec_model.train(
            tagged_docs,
            total_examples=self.doc2vec_model.corpus_count,
            epochs=self.doc2vec_model.epochs
        )

        print(f"Doc2Vec model trained with vocabulary size: {len(self.doc2vec_model.wv)}")

        return self.doc2vec_model

    def transform_doc2vec(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to Doc2Vec vectors

        Args:
            texts: List of text documents

        Returns:
            Document vector matrix
        """
        if self.doc2vec_model is None:
            raise ValueError("Doc2Vec model not trained yet")

        vectors = [
            self.doc2vec_model.infer_vector(text.split())
            for text in texts
        ]

        return np.array(vectors)

    def save_vectorizers(self, output_dir: str):
        """
        Save all fitted vectorizers

        Args:
            output_dir: Output directory path
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.count_vectorizer:
            with open(output_path / 'count_vectorizer.pkl', 'wb') as f:
                pickle.dump(self.count_vectorizer, f)
            print(f"Count Vectorizer saved")

        if self.tfidf_vectorizer:
            with open(output_path / 'tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            print(f"TF-IDF Vectorizer saved")

        if self.word2vec_model:
            self.word2vec_model.save(str(output_path / 'word2vec_model.bin'))
            print(f"Word2Vec model saved")

        if self.doc2vec_model:
            self.doc2vec_model.save(str(output_path / 'doc2vec_model.bin'))
            print(f"Doc2Vec model saved")

    def load_vectorizers(self, input_dir: str):
        """
        Load all vectorizers

        Args:
            input_dir: Input directory path
        """
        from pathlib import Path

        input_path = Path(input_dir)

        count_vec_path = input_path / 'count_vectorizer.pkl'
        if count_vec_path.exists():
            with open(count_vec_path, 'rb') as f:
                self.count_vectorizer = pickle.load(f)
            print(f"Count Vectorizer loaded")

        tfidf_vec_path = input_path / 'tfidf_vectorizer.pkl'
        if tfidf_vec_path.exists():
            with open(tfidf_vec_path, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            print(f"TF-IDF Vectorizer loaded")

        w2v_path = input_path / 'word2vec_model.bin'
        if w2v_path.exists():
            self.word2vec_model = Word2Vec.load(str(w2v_path))
            print(f"Word2Vec model loaded")

        d2v_path = input_path / 'doc2vec_model.bin'
        if d2v_path.exists():
            self.doc2vec_model = Doc2Vec.load(str(d2v_path))
            print(f"Doc2Vec model loaded")


if __name__ == "__main__":
    # Example usage
    vectorizer = TextVectorizer()

    sample_texts = [
        "email application not working",
        "printer issue paper jam",
        "vpn connection problem",
        "password reset needed",
        "application crash error"
    ]

    # Count Vectorizer
    vectorizer.fit_count_vectorizer(sample_texts, max_features=50)
    count_vectors = vectorizer.transform_count_vectorizer(sample_texts)
    print(f"Count vectors shape: {count_vectors.shape}")

    # TF-IDF Vectorizer
    vectorizer.fit_tfidf_vectorizer(sample_texts, max_features=50)
    tfidf_vectors = vectorizer.transform_tfidf_vectorizer(sample_texts)
    print(f"TF-IDF vectors shape: {tfidf_vectors.shape}")

    # Word2Vec
    vectorizer.fit_word2vec(sample_texts, vector_size=50)
    w2v_vectors = vectorizer.transform_word2vec(sample_texts)
    print(f"Word2Vec vectors shape: {w2v_vectors.shape}")
