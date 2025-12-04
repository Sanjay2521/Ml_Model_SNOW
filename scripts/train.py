"""
Training Script
Main script to train all models
"""

import sys
import argparse
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import TextCleaner, DataLoader, remove_duplicates, handle_null_values
from src.features import FeatureExtractor, TextVectorizer
from src.models import TraditionalMLModels, DeepLearningModels, EnsembleModels
from src.evaluation import ModelEvaluator
from src.utils import load_config, setup_logger, create_directories, format_time

import numpy as np
import pandas as pd


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train ML models for incident auto-assignment')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--model', type=str, default='all', help='Specific model to train (or "all")')
    parser.add_argument('--vectorizer', type=str, default='tfidf',
                       choices=['count', 'tfidf', 'word2vec'], help='Vectorization method')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs for deep learning models')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for deep learning models')
    parser.add_argument('--output-dir', type=str, default='models/saved_models/', help='Output directory for models')

    return parser.parse_args()


def main():
    """Main training function"""
    start_time = time.time()

    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logger
    logger = setup_logger(
        name='training',
        log_file=config.get('output', {}).get('logs_path', 'results/logs/') + 'training.log',
        level=config.get('logging', {}).get('level', 'INFO')
    )

    logger.info("="*60)
    logger.info("Starting Training Pipeline")
    logger.info("="*60)

    # Create directories
    output_dirs = [
        config.get('output', {}).get('models_path', 'models/saved_models/'),
        config.get('output', {}).get('plots_path', 'results/plots/'),
        config.get('output', {}).get('reports_path', 'results/reports/'),
        config.get('data', {}).get('processed_path', 'data/processed/')
    ]
    create_directories(output_dirs)

    # Step 1: Load Data
    logger.info("\n[Step 1] Loading Data...")
    data_loader = DataLoader(args.config)

    csv_path = Path(config['data']['raw_path']) / config['data']['incident_csv']

    if csv_path.exists():
        df = data_loader.load_csv(str(csv_path))
    else:
        logger.warning(f"Data file not found: {csv_path}")
        logger.info("Using sample data for demonstration...")
        from src.preprocessing import load_sample_data
        df = load_sample_data()

    data_loader.print_data_summary(df, "Initial Dataset")

    # Step 2: Data Preprocessing
    logger.info("\n[Step 2] Data Preprocessing...")

    # Handle null values and duplicates
    df = handle_null_values(df, strategy='fill')
    df = remove_duplicates(df)

    # Initialize text cleaner
    text_cleaner = TextCleaner(config.get('preprocessing', {}))

    # Get text columns
    text_columns = config.get('data', {}).get('text_columns', ['short_description', 'description'])
    existing_text_cols = [col for col in text_columns if col in df.columns]

    if existing_text_cols:
        df = text_cleaner.clean_dataframe(df, existing_text_cols)
    else:
        # Create dummy cleaned_text if no text columns found
        logger.warning("No text columns found, creating dummy cleaned_text")
        df['cleaned_text'] = df.iloc[:, 0].astype(str)

    # Save processed data
    processed_path = config['data']['processed_path'] + 'processed_data.csv'
    data_loader.save_processed_data(df, 'processed_data.csv')

    # Step 3: Feature Engineering
    logger.info("\n[Step 3] Feature Engineering...")

    # Initialize feature extractor
    feature_extractor = FeatureExtractor(config)

    # Extract additional features
    categorical_cols = ['priority', 'impact', 'urgency', 'category']
    existing_cat_cols = [col for col in categorical_cols if col in df.columns]

    if existing_cat_cols:
        df = feature_extractor.extract_all_features(df, 'cleaned_text', existing_cat_cols)

    # Initialize vectorizer
    vectorizer = TextVectorizer(config)

    # Prepare features and target
    X_text = df['cleaned_text'].tolist()
    target_col = config.get('data', {}).get('target_column', 'assignment_group')

    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in data")
        logger.info(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    y = df[target_col].values

    # Split data
    train_df, val_df, test_df = data_loader.split_data(
        df,
        target_col,
        train_size=config['data']['train_split'],
        val_size=config['data']['val_split'],
        test_size=config['data']['test_split'],
        random_state=config['data']['random_state']
    )

    # Vectorize text
    logger.info(f"\nVectorizing text using {args.vectorizer}...")

    if args.vectorizer == 'count':
        vectorizer.fit_count_vectorizer(train_df['cleaned_text'].tolist())
        X_train = vectorizer.transform_count_vectorizer(train_df['cleaned_text'].tolist())
        X_val = vectorizer.transform_count_vectorizer(val_df['cleaned_text'].tolist())
        X_test = vectorizer.transform_count_vectorizer(test_df['cleaned_text'].tolist())

    elif args.vectorizer == 'tfidf':
        vectorizer.fit_tfidf_vectorizer(train_df['cleaned_text'].tolist())
        X_train = vectorizer.transform_tfidf_vectorizer(train_df['cleaned_text'].tolist())
        X_val = vectorizer.transform_tfidf_vectorizer(val_df['cleaned_text'].tolist())
        X_test = vectorizer.transform_tfidf_vectorizer(test_df['cleaned_text'].tolist())

    elif args.vectorizer == 'word2vec':
        vectorizer.fit_word2vec(train_df['cleaned_text'].tolist())
        X_train = vectorizer.transform_word2vec(train_df['cleaned_text'].tolist())
        X_val = vectorizer.transform_word2vec(val_df['cleaned_text'].tolist())
        X_test = vectorizer.transform_word2vec(test_df['cleaned_text'].tolist())

    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values

    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Validation set shape: {X_val.shape}")
    logger.info(f"Test set shape: {X_test.shape}")

    # Save vectorizer
    vectorizer.save_vectorizers(config['output']['models_path'])
    feature_extractor.save_encoders(config['output']['models_path'] + 'label_encoders.pkl')

    # Step 4: Train Models
    logger.info("\n[Step 4] Training Models...")

    # Initialize evaluator
    evaluator = ModelEvaluator(config)

    # Train Traditional ML Models
    if args.model == 'all' or args.model in ['logistic_regression', 'random_forest', 'svm', 'naive_bayes', 'knn', 'decision_tree', 'gradient_boosting', 'sgd']:
        logger.info("\nTraining Traditional ML Models...")
        ml_models = TraditionalMLModels(config)

        models_to_train = config.get('models', {}).get('traditional', {})

        if args.model != 'all':
            models_to_train = {args.model: models_to_train.get(args.model, {'enabled': True})}

        for model_name, model_config in models_to_train.items():
            if model_config.get('enabled', True):
                try:
                    # Train model
                    model = ml_models.train_model(model_name, X_train, y_train)

                    # Predictions
                    y_pred = ml_models.predict(model_name, X_test)
                    y_pred_proba = ml_models.predict_proba(model_name, X_test) if hasattr(model, 'predict_proba') else None

                    # Evaluate
                    metrics = evaluator.calculate_all_metrics(y_test, y_pred, y_pred_proba, model_name)
                    evaluator.print_metrics(model_name)

                    # Save model
                    model_path = config['output']['models_path'] + f'{model_name}.pkl'
                    ml_models.save_model(model_name, model_path)

                except Exception as e:
                    logger.error(f"Error training {model_name}: {str(e)}")

    # Step 5: Model Comparison
    logger.info("\n[Step 5] Model Comparison...")
    comparison_df = evaluator.compare_models(config['output']['reports_path'] + 'model_comparison.csv')

    # Plot comparison
    evaluator.plot_model_comparison(
        save_path=config['output']['plots_path'] + 'model_comparison.png'
    )

    # Save evaluation results
    evaluator.save_results(config['output']['reports_path'] + 'evaluation_results.json')

    # Training complete
    end_time = time.time()
    total_time = end_time - start_time

    logger.info("\n" + "="*60)
    logger.info(f"Training Complete!")
    logger.info(f"Total Time: {format_time(total_time)}")
    logger.info(f"Models saved to: {config['output']['models_path']}")
    logger.info(f"Results saved to: {config['output']['reports_path']}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
