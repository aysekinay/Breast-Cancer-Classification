# Breast Cancer Classification with Machine Learning

## Project Overview
This academic team project uses the Breast Cancer Wisconsin dataset from scikit-learn to compare machine learning classification models and preprocessing techniques.

## Models
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

## Data Preprocessing and Experiments
1. Baseline models using the original dataset.
2. Mean imputation after introducing approximately 5% missing values.
3. Feature selection using Variance Threshold.
4. Recursive Feature Elimination (RFE) with feature scaling using StandardScaler.

## Evaluation Metrics
- Accuracy
- F1-score
- ROC-AUC
- 5-fold cross-validation mean and variance

## Key Results
The best reported results were achieved by Logistic Regression with RFE and feature scaling:
- Accuracy: 98.2%
- F1-score: 98.6%
- ROC-AUC: 98.1%

SVM also achieved 98.2% accuracy with RFE and scaling.

## Technologies
- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

## Teamwork
This project was completed collaboratively by two students as an academic assignment.

## Installation
Make sure Python is installed. Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## How to Run
Run the scripts from the project folder:

```bash
python project_fs1.py
python project_fs2.py
```

## Dataset
The project uses the Breast Cancer Wisconsin dataset provided by scikit-learn.

## Disclaimer
This is an educational machine learning project, not a medical diagnostic tool.
