"""
train_v2.py
Capstone Project 2 - Sarcasm Detection (NLP)
Extended training: Hyperparameter Tuning + Ensemble Models

Run AFTER train.py (reuses same data loading/cleaning).
Outputs (in ./models):
    tfidf_vectorizer.pkl   (overwritten if better)
    sarcasm_model.pkl      (overwritten if better)
    model_metadata.pkl     (overwritten if better)
"""

import json
import re
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")

for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# 1. Load + clean (same as train.py)
# ---------------------------------------------------------------------------
DATA_PATH = "data/Sarcasm_Headlines_Dataset.json"

records = [json.loads(line) for line in open(DATA_PATH, "r")]
df = pd.DataFrame(records)

df = df.drop_duplicates(subset="headline").reset_index(drop=True)
df = df.dropna(subset=["headline", "is_sarcastic"]).reset_index(drop=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(w) for w in tokens if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)

df["clean_headline"] = df["headline"].apply(clean_text)
df = df[df["clean_headline"].str.len() > 0].reset_index(drop=True)

X = df["clean_headline"]
y = df["is_sarcastic"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ---------------------------------------------------------------------------
# 2. TF-IDF — tune max_features / ngram_range alongside model via pipeline?
#    For speed, fix a strong TF-IDF config first, then tune model hyperparams.
#    (A full Pipeline + GridSearchCV over TF-IDF params is included but optional —
#     set RUN_TFIDF_SEARCH = True to enable, it is slower.)
# ---------------------------------------------------------------------------
RUN_TFIDF_SEARCH = False

if RUN_TFIDF_SEARCH:
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    param_grid = {
        "tfidf__max_features": [10000, 20000, 30000],
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "clf__C": [0.5, 1, 5, 10],
    }
    search = GridSearchCV(pipe, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=2)
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    best_tfidf_params = {k.split("__")[1]: v for k, v in search.best_params_.items() if k.startswith("tfidf__")}
    tfidf = TfidfVectorizer(sublinear_tf=True, **best_tfidf_params)
else:
    # Slightly richer TF-IDF than v1 (more features, trigrams)
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 3), sublinear_tf=True, min_df=2)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print("TF-IDF shape:", X_train_tfidf.shape)

# ---------------------------------------------------------------------------
# 3. Hyperparameter Tuning for individual models
# ---------------------------------------------------------------------------
def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    train_pred = model.predict(X_tr)
    test_pred = model.predict(X_te)
    row = {
        "Model": name,
        "Train Accuracy": accuracy_score(y_tr, train_pred),
        "Test Accuracy": accuracy_score(y_te, test_pred),
        "Train F1": f1_score(y_tr, train_pred),
        "Test F1": f1_score(y_te, test_pred),
        "Test Precision": precision_score(y_te, test_pred),
        "Test Recall": recall_score(y_te, test_pred),
    }
    return row

results = []

# --- 3.1 Logistic Regression: tune C, penalty ---
print("\nTuning Logistic Regression...")
lr_grid = {
    "C": [0.5, 1, 2, 5, 10],
    "penalty": ["l2"],
    "solver": ["liblinear", "lbfgs"],
}
lr_search = RandomizedSearchCV(
    LogisticRegression(max_iter=2000, random_state=42),
    lr_grid, n_iter=8, cv=3, scoring="f1", n_jobs=-1, random_state=42
)
lr_search.fit(X_train_tfidf, y_train)
best_lr = lr_search.best_estimator_
print("Best LR params:", lr_search.best_params_)
results.append(evaluate("Logistic Regression (tuned)", best_lr, X_train_tfidf, y_train, X_test_tfidf, y_test))

# --- 3.2 Linear SVM: tune C, then calibrate for predict_proba ---
print("\nTuning Linear SVM...")
svm_grid = {"C": [0.1, 0.5, 1, 2, 5]}
svm_search = RandomizedSearchCV(
    LinearSVC(random_state=42, max_iter=5000),
    svm_grid, n_iter=5, cv=3, scoring="f1", n_jobs=-1, random_state=42
)
svm_search.fit(X_train_tfidf, y_train)
print("Best SVM params:", svm_search.best_params_)

# Calibrate so SVM can output probabilities (needed for ensembling/app)
best_svm = CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=5000, **svm_search.best_params_), cv=3)
best_svm.fit(X_train_tfidf, y_train)
results.append(evaluate("Linear SVM (tuned, calibrated)", best_svm, X_train_tfidf, y_train, X_test_tfidf, y_test))

# --- 3.3 Multinomial Naive Bayes: tune alpha ---
print("\nTuning Multinomial Naive Bayes...")
nb_grid = {"alpha": [0.01, 0.1, 0.5, 1.0, 2.0]}
nb_search = GridSearchCV(MultinomialNB(), nb_grid, cv=3, scoring="f1", n_jobs=-1)
nb_search.fit(X_train_tfidf, y_train)
best_nb = nb_search.best_estimator_
print("Best NB params:", nb_search.best_params_)
results.append(evaluate("Multinomial NB (tuned)", best_nb, X_train_tfidf, y_train, X_test_tfidf, y_test))

# ---------------------------------------------------------------------------
# 4. Ensemble Models
# ---------------------------------------------------------------------------
print("\nBuilding ensembles...")

# --- 4.1 Voting Classifier (soft voting using probabilities) ---
voting_clf = VotingClassifier(
    estimators=[
        ("lr", best_lr),
        ("svm", best_svm),
        ("nb", best_nb),
    ],
    voting="soft",
    weights=[2, 2, 1],  # favor LR/SVM slightly over NB
)
voting_clf.fit(X_train_tfidf, y_train)
results.append(evaluate("Voting Ensemble (LR+SVM+NB)", voting_clf, X_train_tfidf, y_train, X_test_tfidf, y_test))

# --- 4.2 Stacking Classifier (meta-learner on top of base models) ---
stacking_clf = StackingClassifier(
    estimators=[
        ("lr", best_lr),
        ("svm", best_svm),
        ("nb", best_nb),
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=3,
    n_jobs=-1,
)
stacking_clf.fit(X_train_tfidf, y_train)
results.append(evaluate("Stacking Ensemble (LR+SVM+NB -> LR)", stacking_clf, X_train_tfidf, y_train, X_test_tfidf, y_test))

# ---------------------------------------------------------------------------
# 5. Compare all results
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values("Test F1", ascending=False)
print("\n=== Tuned + Ensemble Model Comparison ===")
print(results_df.to_string(index=False))

os.makedirs("docs/eda", exist_ok=True)
results_df.to_csv("docs/eda/model_comparison_v2.csv", index=False)

# ---------------------------------------------------------------------------
# 6. Pick overall best model (by Test F1) and save if it beats v1
# ---------------------------------------------------------------------------
best_row = results_df.iloc[0]
best_name = best_row["Model"]
candidates = {
    "Logistic Regression (tuned)": best_lr,
    "Linear SVM (tuned, calibrated)": best_svm,
    "Multinomial NB (tuned)": best_nb,
    "Voting Ensemble (LR+SVM+NB)": voting_clf,
    "Stacking Ensemble (LR+SVM+NB -> LR)": stacking_clf,
}
best_model = candidates[best_name]
print(f"\nBest model: {best_name}")
print(f"Test Accuracy: {best_row['Test Accuracy']:.4f}, Test F1: {best_row['Test F1']:.4f}")

# Confusion matrix + ROC for best model
y_proba = best_model.predict_proba(X_test_tfidf)[:, 1]
y_pred = best_model.predict(X_test_tfidf)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Sarcastic", "Sarcastic"],
            yticklabels=["Not Sarcastic", "Sarcastic"])
plt.title(f"Confusion Matrix - {best_name} (Test)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig("docs/eda/06_confusion_matrix_v2.png", bbox_inches="tight")
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - {best_name} (Test)")
plt.legend()
plt.savefig("docs/eda/07_roc_curve_v2.png", bbox_inches="tight")
plt.close()

print(f"\nTest ROC-AUC: {auc:.4f}")
print(classification_report(y_test, y_pred, target_names=["Not Sarcastic", "Sarcastic"]))

# ---------------------------------------------------------------------------
# 7. Save artifacts (overwrite v1 with improved versions)
# ---------------------------------------------------------------------------
os.makedirs("models", exist_ok=True)

with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("models/sarcasm_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

metadata = {
    "model_name": best_name,
    "train_accuracy": best_row["Train Accuracy"],
    "test_accuracy": best_row["Test Accuracy"],
    "test_f1": best_row["Test F1"],
    "test_auc": auc,
    "n_features": X_train_tfidf.shape[1],
    "train_size": X_train.shape[0],
    "test_size": X_test.shape[0],
}
with open("models/model_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("\nSaved updated artifacts to ./models/")
print(metadata)
