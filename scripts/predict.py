"""
Prediction Script
Make predictions on new incident data
"""

import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import TextCleaner, DataLoader
from src.features import FeatureExtractor, TextVectorizer
from src.utils import load_config
import pandas as pd
import pickle


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Make predictions on incident data')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input', type=str, help='Input CSV file')
    parser.add_argument('--text', type=str, help='Single text to predict')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output file')
    parser.add_argument('--vectorizer', type=str, default='tfidf', choices=['count', 'tfidf', 'word2vec'])

    return parser.parse_args()


def main():
    """Main prediction function"""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    print("="*60)
    print("Incident Auto-Assignment Prediction")
    print("="*60)

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)

    # Load vectorizer
    print(f"Loading vectorizer...")
    vectorizer = TextVectorizer(config)
    vectorizer.load_vectorizers(config['output']['models_path'])

    # Initialize text cleaner
    text_cleaner = TextCleaner(config.get('preprocessing', {}))

    # Prepare input
    if args.text:
        # Single text prediction
        print(f"\nInput text: {args.text}")

        # Clean text
        cleaned_text = text_cleaner.clean_text(args.text)
        print(f"Cleaned text: {cleaned_text}")

        # Vectorize
        if args.vectorizer == 'count':
            X = vectorizer.transform_count_vectorizer([cleaned_text])
        elif args.vectorizer == 'tfidf':
            X = vectorizer.transform_tfidf_vectorizer([cleaned_text])
        elif args.vectorizer == 'word2vec':
            X = vectorizer.transform_word2vec([cleaned_text])

        # Predict
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None

        print(f"\nPredicted Assignment Group: {prediction}")
        if proba is not None:
            print(f"Confidence: {max(proba):.4f}")

    elif args.input:
        # Batch prediction
        print(f"\nLoading data from: {args.input}")
        df = pd.read_csv(args.input)

        print(f"Input shape: {df.shape}")

        # Clean text
        text_columns = config.get('data', {}).get('text_columns', ['short_description', 'description'])
        existing_cols = [col for col in text_columns if col in df.columns]

        if existing_cols:
            df = text_cleaner.clean_dataframe(df, existing_cols)
        else:
            print("No text columns found")
            return

        # Vectorize
        if args.vectorizer == 'count':
            X = vectorizer.transform_count_vectorizer(df['cleaned_text'].tolist())
        elif args.vectorizer == 'tfidf':
            X = vectorizer.transform_tfidf_vectorizer(df['cleaned_text'].tolist())
        elif args.vectorizer == 'word2vec':
            X = vectorizer.transform_word2vec(df['cleaned_text'].tolist())

        # Predict
        predictions = model.predict(X)
        df['predicted_assignment_group'] = predictions

        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X)
            df['confidence'] = probas.max(axis=1)

        # Save predictions
        df.to_csv(args.output, index=False)
        print(f"\nPredictions saved to: {args.output}")
        print(f"Total predictions: {len(predictions)}")

    else:
        print("Error: Please provide either --text or --input")

    print("\n" + "="*60)
    print("Prediction Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
