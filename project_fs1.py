from sklearn.datasets import load_breast_cancer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

df = X.copy()
df["target"] = y

print(df.shape)

print(df["target"].value_counts())

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.show(block=True)
print(df.isnull().sum())

import numpy as np

X_missing = X.copy()

# =====%5====

mask = np.random.rand(*X_missing.shape) < 0.05
X_missing[mask] = np.nan

print("Missing values after adding:")
print(X_missing.isnull().sum())

#=====Imputation=====

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

X_filled = pd.DataFrame(
    imputer.fit_transform(X_missing),
    columns=X.columns
)

#===VARIANCE THRESHOLD====

from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)

X_selected = selector.fit_transform(X_filled)

X_selected = pd.DataFrame(X_selected)

#====Train/Test=====


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)


#====LOGISTIC====

model_lr = LogisticRegression(max_iter=5000)

model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

print("\nLogistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("F1:", f1_score(y_test, y_pred_lr))
print("ROC:", roc_auc_score(y_test, y_pred_lr))

#====  K-FOLD =====

scores_lr = cross_val_score(model_lr, X_selected, y, cv=5)

print("LR CV:", scores_lr)
print("Mean:", scores_lr.mean())
print("Variance:", scores_lr.var())

#==== SVM ====

model_svm = SVC(probability=True)

model_svm.fit(X_train, y_train)
y_pred_svm = model_svm.predict(X_test)

print("\nSVM:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("F1:", f1_score(y_test, y_pred_svm))
print("ROC:", roc_auc_score(y_test, y_pred_svm))

#==== K-FOLD =====
scores_svm = cross_val_score(model_svm, X_selected, y, cv=5)

print("SVM CV:", scores_svm)
print("Mean:", scores_svm.mean())
print("Variance:", scores_svm.var())

#==== Random Forest ====

from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(random_state=42)

model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)

print("\nRandom Forest:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("F1:", f1_score(y_test, y_pred_rf))
print("ROC:", roc_auc_score(y_test, y_pred_rf))

#=== K-FOLD ===

scores_rf = cross_val_score(model_rf, X_selected, y, cv=5)

print("RF CV:", scores_rf)
print("Mean:", scores_rf.mean())
print("Variance:", scores_rf.var())

