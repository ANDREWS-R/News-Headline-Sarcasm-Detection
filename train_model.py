import re
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

DATASET_PATH = "dataset/Sarcasm_Headlines_Dataset.json"

data = []

with open(DATASET_PATH, "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line))

data = pd.DataFrame(data)

print("Original dataset shape:", data.shape)


# --------------------------------------------------
# Remove Duplicate Records
# --------------------------------------------------

data = data.drop_duplicates()

data = data.drop_duplicates(
    subset=["headline", "is_sarcastic"]
)

print("Dataset shape after removing duplicates:", data.shape)


# --------------------------------------------------
# Text Preprocessing
# --------------------------------------------------

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


data["clean_headline"] = data["headline"].apply(
    preprocess_text
)


# --------------------------------------------------
# Separate Features and Labels
# --------------------------------------------------

X = data["clean_headline"]
y = data["is_sarcastic"]


# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# TF-IDF Feature Extraction
# --------------------------------------------------

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

print("Number of TF-IDF features:", X_train_tfidf.shape[1])


# --------------------------------------------------
# Model Comparison
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC()
}

results = []


for name, model in models.items():

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    results.append(
        [name, accuracy, precision, recall, f1]
    )

    print("\n", name)
    print("Accuracy :", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall   :", round(recall * 100, 2), "%")
    print("F1 Score :", round(f1 * 100, 2), "%")


# --------------------------------------------------
# Select Final Model
# --------------------------------------------------

svm_model = models["Linear SVM"]

print("\nFinal model selected: Linear SVM")


# --------------------------------------------------
# Save Model and Vectorizer
# --------------------------------------------------

joblib.dump(
    svm_model,
    "sarcasm_svm_model.pkl"
)

joblib.dump(
    tfidf,
    "tfidf_vectorizer.pkl"
)

print("\nModel saved as: sarcasm_svm_model.pkl")
print("Vectorizer saved as: tfidf_vectorizer.pkl")


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

svm_predictions = svm_model.predict(X_test_tfidf)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    svm_predictions,
    display_labels=["Non-Sarcastic", "Sarcastic"]
)

plt.title("Linear SVM - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()