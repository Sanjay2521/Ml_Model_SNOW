# ServiceNow Incident Auto-Assignment ML Model

## 🎯 Project Overview

This is a comprehensive end-to-end Machine Learning solution for automatically assigning ServiceNow incidents to the appropriate team/personnel based on incident details. The system uses NLP and ML techniques to analyze incident descriptions and predict the best assignment group.

## 📋 Features

### Data Preprocessing
- Drop irrelevant columns
- Handle null values and duplicates
- Text cleaning (lowercase conversion, special characters removal)
- Remove common words, stop words, URLs, email IDs, phone numbers
- Remove file paths and normalize spacing

### Feature Engineering
- Tokenization
- Count Vectorizer
- TF-IDF Vectorizer
- Word2Vec embeddings
- Label Encoding
- Named Entity Recognition (NER)
- POS Tagging

### Machine Learning Models
**Traditional ML:**
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gradient Boosting
- Naive Bayes
- Decision Tree
- Random Forest
- Logistic Regression
- Stochastic Gradient Descent (SGD)

**Deep Learning:**
- Deep Neural Network (DNN)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Level 1 / Level 2 Modeling (Ensemble)

### Model Evaluation
- Accuracy
- F1 Score (Precision & Recall)
- Cohen Kappa Score
- Loss metrics
- Hyperparameter tuning
- Ensemble methods
- Model comparison charts

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Ml_Model_SNOW
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data:**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
```

## 📁 Project Structure

```
Ml_Model_SNOW/
├── data/
│   ├── raw/                          # Place your raw datasets here
│   ├── processed/                    # Processed datasets
│   └── sample/                       # Sample data for testing
├── notebooks/
│   ├── 01_EDA.ipynb                 # Exploratory Data Analysis
│   ├── 02_Data_Preprocessing.ipynb  # Data cleaning and preprocessing
│   ├── 03_Feature_Engineering.ipynb # Feature engineering experiments
│   ├── 04_Model_Training.ipynb      # Model training and evaluation
│   └── 05_Model_Comparison.ipynb    # Compare all models
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── text_cleaner.py          # Text cleaning functions
│   │   └── data_loader.py           # Data loading utilities
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_extractor.py     # Feature engineering
│   │   └── vectorizers.py           # Vectorization methods
│   ├── models/
│   │   ├── __init__.py
│   │   ├── traditional_ml.py        # Traditional ML models
│   │   ├── deep_learning.py         # Deep learning models
│   │   └── ensemble.py              # Ensemble methods
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py               # Evaluation metrics
│   └── utils/
│       ├── __init__.py
│       ├── config.py                # Configuration
│       └── helpers.py               # Helper functions
├── models/
│   ├── saved_models/                # Trained models
│   └── checkpoints/                 # Model checkpoints
├── results/
│   ├── plots/                       # Visualization plots
│   ├── reports/                     # Evaluation reports
│   └── logs/                        # Training logs
├── scripts/
│   ├── train.py                     # Training script
│   ├── predict.py                   # Prediction script
│   ├── evaluate.py                  # Evaluation script
│   └── deploy.py                    # Deployment script
├── tests/
│   └── test_preprocessing.py        # Unit tests
├── requirements.txt                  # Python dependencies
├── setup.py                         # Package setup
├── config.yaml                      # Configuration file
└── README.md                        # This file
```

## 💻 Usage

### 1. Place Your Data
Put your datasets in the `data/raw/` folder:
- `AnyConv.com__incident (1).csv` - Historical incidents data
- `AMS_ACC_Incident_KPIs (1).xlsx` - Incident KPIs

### 2. Run Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### 3. Train Models
```bash
# Train all models
python scripts/train.py --config config.yaml

# Train specific model
python scripts/train.py --model random_forest

# Train with custom parameters
python scripts/train.py --model lstm --epochs 50 --batch-size 32
```

### 4. Evaluate Models
```bash
python scripts/evaluate.py --model-path models/saved_models/best_model.pkl
```

### 5. Make Predictions
```bash
# Predict from file
python scripts/predict.py --input data/new_incidents.csv --output predictions.csv

# Single prediction
python scripts/predict.py --text "Unable to access email application"
```

## 📊 Model Performance

Results will be saved in `results/reports/model_comparison.csv` with metrics for all models.

## 🔧 Configuration

Edit `config.yaml` to customize:
- Data paths
- Model hyperparameters
- Feature engineering options
- Training parameters

## 🐳 Docker Support (Optional)

```bash
docker build -t ml-snow-incident .
docker run -p 5000:5000 ml-snow-incident
```

## 📝 API Documentation

After training, you can deploy the model as an API:

```bash
python scripts/deploy.py
```

API will be available at `http://localhost:5000`

**Endpoints:**
- `POST /predict` - Make predictions
- `GET /health` - Health check
- `GET /metrics` - Model metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

MIT License

## 👥 Authors

- Your Name

## 🙏 Acknowledgments

- ServiceNow for the use case
- Open source ML community

## 📞 Support

For issues and questions, please open an issue on GitHub.

---

**Last Updated:** December 2025
