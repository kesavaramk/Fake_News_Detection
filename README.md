# 🔍 AI Fake News Detector

> Detect misinformation using NLP and Machine Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- ✅ Real-time single article analysis
- 📂 Batch CSV processing
- 🔍 Explainable AI (highlighted keywords + top contributing words)
- 📊 Confidence scoring with uncertainty zone
- 🌐 URL scraping for direct article analysis
- 🎯 Ensemble ML model (LR + SVM + Random Forest)
- 🚀 Auto-trains on first run

## Installation

```bash
git clone https://github.com/your-username/Fake_News_Detection.git
cd Fake_News_Detection
pip install -r requirements.txt
```

## Quick Start

```bash
# Run the app (auto-trains model on first run)
streamlit run app/main.py

# Or train manually first
python scripts/train.py

# Then launch app
streamlit run app/main.py
```

## Project Architecture

```
User Input (Text/URL)
        ↓
┌─────────────────────┐
│   Text Preprocessor  │  clean, tokenize, lemmatize
└─────────┬───────────┘
          ↓
┌─────────────────────┐
│  Feature Extractor   │  TF-IDF (word+char) + linguistic
└─────────┬───────────┘
          ↓
┌────────────────────────────────┐
│         Ensemble Model          │
│  LR (40%) + SVM (35%) + RF(25%) │
└─────────┬──────────────────────┘
          ↓
┌─────────────────────┐
│    Explainability    │  top words, highlighted HTML
└─────────┬───────────┘
          ↓
    Result + Confidence
```

## Model Performance

| Metric | Value |
|--------|-------|
| Training samples | 2,000 synthetic articles |
| Accuracy | ~90%+ on training data |
| Features | TF-IDF (word + character n-grams) + 8 linguistic features |

## Project Structure

```
Fake_News_Detection/
├── app/                # Streamlit application
│   └── main.py
├── data/
│   └── sample/         # Sample datasets
├── models/             # Saved model artifacts
├── notebooks/          # EDA and experiments
│   └── EDA.ipynb
├── scripts/
│   └── train.py        # Model training script
├── src/                # Core library modules
│   ├── preprocessor.py
│   ├── features.py
│   └── model.py
├── tests/              # Unit tests
├── requirements.txt
└── README.md
```

## Tech Stack

| Component | Library |
|-----------|---------|
| UI | Streamlit |
| ML Models | scikit-learn |
| NLP | NLTK |
| Charts | Plotly |
| Web Scraping | BeautifulSoup4 |
| Data | pandas, numpy |

## Disclaimer

> ⚠️ This tool is for **educational purposes only**. Always verify news from multiple trusted sources. AI is not infallible.
