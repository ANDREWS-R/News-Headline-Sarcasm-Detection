# News Headline Sarcasm Detection Using NLP and Machine Learning

## 1. Project Overview

Sarcasm is a form of communication in which the intended meaning of a statement may differ from its literal meaning. Detecting sarcasm in written text is challenging because the system must identify linguistic patterns that may indicate an implied or opposite meaning.

This project develops a Natural Language Processing (NLP) based system that classifies news headlines as either **Sarcastic** or **Non-Sarcastic**.

The system uses text preprocessing, TF-IDF feature extraction, and machine learning classification techniques. Three classification models were evaluated:

- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (SVM)

Based on the evaluation results, Linear SVM was selected as the final model.

---

## 2. Motivation and Accessibility

The motivation for this project comes from an interest in making digital spaces more accessible and understandable for a wider range of users.

Sarcasm can be difficult to interpret from text because the literal meaning of a sentence may not represent the intended meaning. Some users may find implied meanings, indirect language, or sarcasm more difficult to interpret.

This project explores how NLP can be used as an accessibility-oriented assistive technology by identifying potentially sarcastic content.

The current implementation focuses on news headlines as a manageable first step. The broader vision is to extend the system into a browser extension that could analyse web content in real time and indicate potentially sarcastic statements.

Such a system could provide an additional layer of contextual assistance while users browse online content.

---

## 3. Problem Statement

To develop an NLP-based machine learning system capable of automatically classifying news headlines as sarcastic or non-sarcastic.

The system accepts a text headline as input, processes the text, converts it into numerical features, and generates a classification result.

---

## 4. Objectives

The main objectives of this project are:

1. To study sarcasm detection as an NLP classification problem.
2. To preprocess news headline text for machine learning.
3. To convert textual data into numerical representations using TF-IDF.
4. To compare multiple machine learning classification algorithms.
5. To select the best-performing model based on evaluation metrics.
6. To develop a simple Python-based application for sarcasm prediction.
7. To explore the potential of sarcasm detection as an accessibility-oriented technology.
8. To identify possible improvements and future applications such as real-time web-content analysis.

---

## 5. Dataset

### Dataset Name

**News Headlines Dataset for Sarcasm Detection**

### Dataset Source

The dataset was created by Rishabh Misra and is available on GitHub:

[https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection](https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection)

### Dataset Size

The original dataset contains **28,619 news headlines**.

After removing duplicate records, the final dataset contains:

**28,503 records**

### Attributes

| Attribute | Description |
|---|---|
| `is_sarcastic` | Target label indicating whether the headline is sarcastic |
| `headline` | News headline text |
| `article_link` | Link associated with the original article |

### Classes

The dataset contains two classes:

- `0` — Non-Sarcastic
- `1` — Sarcastic

### Data Preparation

Duplicate records were removed before training.

Text preprocessing included:

- Converting text to lowercase
- Removing URLs
- Normalizing whitespace

Punctuation and stopwords were retained because wording and sentence structure can contribute to the interpretation of sarcasm.

---

## 6. Methodology

The overall workflow of the system is:

```text
News Headlines Dataset
          ↓
Duplicate Removal
          ↓
Text Preprocessing
          ↓
Train-Test Split
          ↓
TF-IDF Feature Extraction
          ↓
Machine Learning Models
          ↓
Model Comparison
          ↓
Linear SVM Selection
          ↓
Prediction
          ↓
Evaluation
```

### 6.1 Text Preprocessing

Each headline is converted to lowercase.

URLs are removed using regular expressions and unnecessary whitespace is normalized.

The preprocessing is intentionally conservative because excessive removal of words or punctuation may remove useful linguistic information.

### 6.2 Train-Test Split

The processed dataset was divided into:

- 80% training data
- 20% testing data

A fixed random state and stratified splitting were used to ensure reproducible and balanced evaluation.

The final split contained:

- 22,802 training samples
- 5,701 testing samples

### 6.3 TF-IDF Feature Extraction

Term Frequency-Inverse Document Frequency (TF-IDF) was used to convert the text headlines into numerical feature vectors.

Both unigrams and bigrams were used:

```
ngram_range = (1, 2)
```

Terms appearing in fewer than two documents were excluded:

```
min_df = 2
```

The resulting feature representation contained **33,967 features**.

### 6.4 Machine Learning Models

Three models were trained and compared:

- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

### 6.5 Final Model

Linear SVM was selected because it achieved the highest overall accuracy and F1-score among the evaluated models.

---

## 7. Results

The models produced the following results on the test set:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 84.74% | 83.05% | 85.32% | 84.17% |
| Naive Bayes | 84.90% | 86.10% | 81.37% | 83.67% |
| **Linear SVM** | **85.14%** | **83.72%** | **85.36%** | **84.53%** |

### Final Model Performance — Linear SVM

- **Accuracy:** 85.14%
- **Precision:** 83.72%
- **Recall:** 85.36%
- **F1-Score:** 84.53%

### Confusion Matrix

The final Linear SVM model produced the following confusion matrix:

| | Predicted Non-Sarcastic | Predicted Sarcastic |
|---|---|---|
| **Actual Non-Sarcastic** | 2540 | 450 |
| **Actual Sarcastic** | 397 | 2314 |

The model correctly classified 2540 non-sarcastic headlines and 2314 sarcastic headlines.

### Sample Application Output

**Example input:**

> Government solves traffic problem by asking everyone to stay home

**Prediction:** `Sarcastic`

---

## 8. Application

A Streamlit-based Python application was developed as the user-facing component of the project.

The application:

- Accepts a news headline from the user.
- Applies the same preprocessing used during training.
- Converts the headline into TF-IDF features.
- Passes the features to the trained Linear SVM model.
- Displays the predicted class.

The application uses the saved trained model and TF-IDF vectorizer.

---

## 9. Project Structure

```
News-Headline-Sarcasm-Detection/
│
├── app.py
├── train_model.py
├── requirements.txt
├── .gitignore
│
├── sarcasm_svm_model.pkl
├── tfidf_vectorizer.pkl
│
├── dataset/
│   └── Sarcasm_Headlines_Dataset.json
│
├── screenshots/
│   ├── application.png
│   ├── model_comparison.png
│   └── confusion_matrix.png
│
└── report/
    └── NLP_Project_Report.pdf
```

---

## 10. Technologies Used

**Programming Language**
- Python

**Libraries**
- Pandas
- Scikit-learn
- Joblib
- Matplotlib
- Streamlit

**NLP Techniques**
- Text preprocessing
- TF-IDF feature extraction
- Unigram and bigram representation

**Machine Learning**
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM

---

## 11. How to Run the Project

**Step 1: Install the required libraries**

```bash
python -m pip install -r requirements.txt
```

**Step 2: Run the application**

```bash
python -m streamlit run app.py
```

**Step 3: Open the application**

Streamlit will provide a local address, normally:

```
http://localhost:8501
```

Enter a news headline and click **Predict**.

---

## 12. Limitations

The current system has several limitations:

- Sarcasm can depend heavily on context, which may not be available in a single headline.
- The model is trained specifically on news headlines and may not perform equally well on social-media posts, comments, conversations, or other forms of web content.
- Machine learning predictions may be incorrect for unfamiliar or ambiguous wording.
- The current system performs binary classification and does not explain why a particular headline was classified as sarcastic.
- The current implementation processes individual headlines rather than analysing complete web pages in real time.
- The model uses traditional machine learning and TF-IDF rather than advanced contextual language models.

---

## 13. Future Scope

The project can be extended in several directions.

### Real-Time Web Browser Extension

The primary future direction is to develop a browser extension that can analyse web content in real time.

A possible workflow would be:

```text
Web Page
   ↓
Extract Text
   ↓
Identify Relevant Sentences
   ↓
NLP Preprocessing
   ↓
Sarcasm Detection Model
   ↓
Accessibility-Oriented Indication
```

This could extend the current news-headline prototype to articles, comments, and other online content.

### Other Improvements

Future improvements could include:

- Larger and more diverse datasets
- Advanced NLP models
- Transformer-based models such as BERT
- Multilingual sarcasm detection
- Context-aware sarcasm detection
- Real-time web-content analysis
- Browser extension integration
- Improved evaluation on different types of online content
- Explainable predictions
- User-configurable accessibility features

---

## 14. Conclusion

This project developed a Python-based NLP application for detecting sarcasm in news headlines.

The system uses text preprocessing, TF-IDF feature extraction, and machine learning classification. Logistic Regression, Multinomial Naive Bayes, and Linear SVM were evaluated, with Linear SVM achieving the best overall performance with an accuracy of 85.14% and an F1-score of 84.53%.

Beyond classification, the project explores the use of NLP as an accessibility-oriented technology. The current news-headline detector serves as a prototype for a broader concept in which sarcasm detection could be integrated into web browsing to provide additional contextual assistance.

The project demonstrates how a relatively simple and explainable NLP pipeline can serve as a foundation for future development toward real-time, accessibility-focused web technologies.