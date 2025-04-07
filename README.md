# Lead Score Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 🚀 Machine Learning system to predict the likelihood of leads converting to internet plan customers.

<!-- 
To add a banner image:
1. Add your image to the 'images' directory
2. Replace this comment with the image markdown:
![Project Banner](images/your-banner-image.png)
-->

## Overview

This project was developed as a final thesis for the Postgraduate Program in Data Science and Big Data at PUC. It aims to identify and assign a probability score to leads receptive to purchasing internet plans using Data Science and Machine Learning techniques.

With this solution, sales, planning, and marketing teams can prioritize leads with higher interest in internet plans.

## 📋 Features

- **Data Collection**: Connection to databases to extract contact information, recharges, and services used by leads
- **Data Processing**: Cleaning, transformation, and preparation of data for modeling
- **Exploratory Analysis**: Generation of visualizations to understand data patterns
- **ML Models**: Implementation of various classification models (Random Forest, SVM, Naive Bayes)
- **Score Generation**: Probability scoring for each lead's conversion potential

## 🧮 ML Models Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Random Forest | 0.89 | 0.87 | 0.93 | 0.90 | 0.92 |
| SVM | 0.85 | 0.84 | 0.88 | 0.86 | 0.89 |
| Logistic Regression | 0.82 | 0.80 | 0.85 | 0.82 | 0.86 |

## 📁 Project Structure

```
leads-score-prediction/
│
├── data/                     # Data files
│   ├── raw/                  # Original raw data
│   └── processed/            # Processed data
│
├── notebooks/                # Jupyter notebooks
│   ├── 01 - Data Collection.ipynb
│   ├── 02 - Data Processing.ipynb
│   ├── 03 - Exploratory Analysis.ipynb
│   └── 04 - ML Models.ipynb
│
├── reports/                  # Generated reports
│   └── figures/              # Visualizations
│
├── models/                   # Trained models
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── data_processing.py    # Data processing pipeline
│   ├── exploratory_analysis.py # Exploratory data analysis
│   ├── modeling.py           # Model training and evaluation
│   ├── predict.py            # Making predictions
│   ├── train.py              # Training pipeline
│   └── utils.py              # Utility functions
│
├── .env.example              # Example environment variables
├── .gitignore                # Git ignore file
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/leads-score-prediction.git
cd leads-score-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your database credentials
```

## 📊 Usage

### Full Pipeline

Run the complete pipeline:
```bash
python main.py
```

### Specific Steps

For individual steps:
```bash
python main.py collect    # Data collection only
python main.py process    # Data processing only
python main.py visualize  # Data visualization only
python main.py train      # Model training only
```

### Predictions

Make predictions with new data:
```bash
python -m src.predict models/best_model.pkl data/raw/new_leads.csv predictions.csv
```

## 📈 Analysis Pipeline

The project follows a predictive analytics pipeline with the following stages:

1. **Data Collection**: Connection to databases to extract leads information
2. **Data Processing**: Cleaning, transformation, and feature engineering
3. **Exploratory Analysis**: Visualization for understanding patterns
4. **Model Training**: Training different classification models
5. **Evaluation**: Performance assessment and model selection
6. **Prediction**: Generating scores for new leads

## 🧪 Technologies Used

- **Data Collection**: Python, SQL, pyodbc
- **Data Processing**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Modeling**: scikit-learn, joblib

## 📊 Data Analysis

This study considers leads across Brazil during the third and fourth quarters of 2021.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

Developed as part of a postgraduate thesis at PUC.

---

<p align="center">
  Made with ❤️ by Adriano Fonseca
</p>

---