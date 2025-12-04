"""
Evaluation Script
Evaluate trained models on test data
"""

import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import TextCleaner, DataLoader
from src.features import TextVectorizer
from src.evaluation import ModelEvaluator
from src.utils import load_config
import pandas as pd
import pickle


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--test-data', type=str, required=True)
    parser.add_argument('--vectorizer', type=str, default='tfidf')
    parser.add_argument('--output-dir', type=str, default='results/')

    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_args()

    config = load_config(args.config)

    print("="*60)
    print("Model Evaluation")
    print("="*60)

    # Load model
    print(f"\nLoading model: {args.model_path}")
    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)

    # Load test data
    print(f"Loading test data: {args.test_data}")
    test_df = pd.read_csv(args.test_data)

    # Load vectorizer
    vectorizer = TextVectorizer(config)
    vectorizer.load_vectorizers(config['output']['models_path'])

    text_cleaner = TextCleaner(config.get('preprocessing', {}))

    # Prepare data
    text_columns = config.get('data', {}).get('text_columns', ['short_description', 'description'])
    existing_cols = [col for col in text_columns if col in test_df.columns]

    if existing_cols:
        test_df = text_cleaner.clean_dataframe(test_df, existing_cols)

    # Vectorize
    if args.vectorizer == 'tfidf':
        X_test = vectorizer.transform_tfidf_vectorizer(test_df['cleaned_text'].tolist())
    elif args.vectorizer == 'count':
        X_test = vectorizer.transform_count_vectorizer(test_df['cleaned_text'].tolist())
    elif args.vectorizer == 'word2vec':
        X_test = vectorizer.transform_word2vec(test_df['cleaned_text'].tolist())

    target_col = config.get('data', {}).get('target_column', 'assignment_group')
    y_test = test_df[target_col].values

    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    # Evaluate
    evaluator = ModelEvaluator(config)
    model_name = Path(args.model_path).stem

    metrics = evaluator.calculate_all_metrics(y_test, y_pred, y_pred_proba, model_name)
    evaluator.print_metrics(model_name)
    evaluator.print_classification_report(y_test, y_pred, model_name=model_name)

    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        y_test, y_pred,
        title=f'Confusion Matrix - {model_name}',
        save_path=f'{args.output_dir}plots/confusion_matrix_{model_name}.png'
    )

    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
