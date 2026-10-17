# Breast Cancer Classification with Machine Learning

## Project Overview
This academic team project explores machine learning classification models using the Breast Cancer Wisconsin dataset from scikit-learn. The goal was to compare model performance under different data preprocessing techniques.

## Models
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

## Data Preprocessing and Experiments
The project compares four approaches:
1. Baseline models using the original dataset.
2. Mean imputation after introducing approximately 5% missing values.
3. Feature selection using Variance Threshold.
4. Recursive Feature Elimination (RFE) combined with feature scaling using StandardScaler.

## Evaluation Metrics
Model performance was evaluated using:
- Accuracy
- F1-score
- ROC-AUC
- 5-fold cross-validation mean and variance

## Key Results
The best reported results were achieved by Logistic Regression with RFE and feature scaling:
- Accuracy: 98.2%
- F1-score: 98.6%
- ROC-AUC: 98.1%

SVM also achieved 98.2% accuracy with RFE and scaling. Random Forest performance was more consistent across the experiments.

## Technologies
- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

## Teamwork
This project was completed collaboratively by two students as an academic assignment.

## Disclaimer
This is an educational machine learning project, not a medical diagnostic tool.

## Installation

Make sure Python is installed on your computer.

## How to Run

Run the scripts from the project folder:

```bash
python project_fs1.py
python project_fs2.py

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Dataset

The project uses the Breast Cancer Wisconsin dataset provided by scikit-learn.

Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
