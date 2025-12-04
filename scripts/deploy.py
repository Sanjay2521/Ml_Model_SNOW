"""
Deployment Script
Deploy model as REST API using Flask
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify
from src.preprocessing import TextCleaner
from src.features import TextVectorizer
from src.utils import load_config
import pickle
import logging

# Initialize Flask app
app = Flask(__name__)

# Global variables
model = None
vectorizer = None
text_cleaner = None
config = None


def load_model_and_config():
    """Load model, vectorizer, and configuration"""
    global model, vectorizer, text_cleaner, config

    # Load configuration
    config = load_config('config.yaml')

    # Load model
    model_path = config.get('api', {}).get('model_path', 'models/saved_models/best_model.pkl')
    print(f"Loading model from: {model_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load vectorizer
    vectorizer = TextVectorizer(config)
    vectorizer.load_vectorizers(config['output']['models_path'])

    # Initialize text cleaner
    text_cleaner = TextCleaner(config.get('preprocessing', {}))

    print("Model and vectorizer loaded successfully!")


@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'service': 'Incident Auto-Assignment API',
        'version': '1.0.0',
        'status': 'running'
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint

    Request JSON format:
    {
        "short_description": "Email not working",
        "description": "Cannot access email application",
        "priority": "2",
        "category": "Email"
    }
    """
    try:
        # Get request data
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Extract text fields
        short_desc = data.get('short_description', '')
        description = data.get('description', '')

        # Combine text
        combined_text = f"{short_desc} {description}"

        if not combined_text.strip():
            return jsonify({'error': 'No text provided'}), 400

        # Clean text
        cleaned_text = text_cleaner.clean_text(combined_text)

        # Vectorize
        X = vectorizer.transform_tfidf_vectorizer([cleaned_text])

        # Predict
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None

        # Prepare response
        response = {
            'predicted_assignment_group': str(prediction),
            'input_text': combined_text[:100] + '...' if len(combined_text) > 100 else combined_text
        }

        if proba is not None:
            response['confidence'] = float(max(proba))
            response['probabilities'] = {
                str(cls): float(prob) for cls, prob in zip(model.classes_, proba)
            }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint

    Request JSON format:
    {
        "incidents": [
            {"short_description": "...", "description": "..."},
            {"short_description": "...", "description": "..."}
        ]
    }
    """
    try:
        data = request.get_json()

        if not data or 'incidents' not in data:
            return jsonify({'error': 'No incidents provided'}), 400

        incidents = data['incidents']

        if not isinstance(incidents, list):
            return jsonify({'error': 'Incidents must be a list'}), 400

        # Process all incidents
        predictions = []

        for incident in incidents:
            short_desc = incident.get('short_description', '')
            description = incident.get('description', '')
            combined_text = f"{short_desc} {description}"

            if combined_text.strip():
                cleaned_text = text_cleaner.clean_text(combined_text)
                X = vectorizer.transform_tfidf_vectorizer([cleaned_text])
                prediction = model.predict(X)[0]

                proba = None
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0]

                pred_result = {
                    'predicted_assignment_group': str(prediction),
                    'confidence': float(max(proba)) if proba is not None else None
                }
            else:
                pred_result = {
                    'predicted_assignment_group': None,
                    'confidence': None,
                    'error': 'No text provided'
                }

            predictions.append(pred_result)

        return jsonify({
            'predictions': predictions,
            'total': len(predictions)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    """Get model metrics"""
    try:
        # Load metrics from file
        import json
        metrics_path = config['output']['reports_path'] + 'evaluation_results.json'

        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)

        return jsonify(metrics_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Load model and configuration
    load_model_and_config()

    # Get API configuration
    host = config.get('api', {}).get('host', '0.0.0.0')
    port = config.get('api', {}).get('port', 5000)
    debug = config.get('api', {}).get('debug', False)

    print(f"\nStarting API server on {host}:{port}")
    print(f"Endpoints:")
    print(f"  - GET  /              : Home")
    print(f"  - GET  /health        : Health check")
    print(f"  - POST /predict       : Single prediction")
    print(f"  - POST /batch_predict : Batch predictions")
    print(f"  - GET  /metrics       : Model metrics")

    # Run Flask app
    app.run(host=host, port=port, debug=debug)
