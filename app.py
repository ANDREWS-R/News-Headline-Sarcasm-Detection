import re
import joblib
import streamlit as st


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="News Headline Sarcasm Detector",
    page_icon="📰",
    layout="centered"
)


# --------------------------------------------------
# Load Trained Model and TF-IDF Vectorizer
# --------------------------------------------------

model = joblib.load("sarcasm_svm_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


# --------------------------------------------------
# Text Preprocessing
# --------------------------------------------------

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_sarcasm(headline):
    cleaned_text = preprocess_text(headline)
    vector = tfidf.transform([cleaned_text])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        return "Sarcastic"
    else:
        return "Non-Sarcastic"


# --------------------------------------------------
# Application Interface
# --------------------------------------------------

st.title("📰 News Headline Sarcasm Detector")

st.write(
    "Enter a news headline below and the machine learning "
    "model will predict whether it is sarcastic or non-sarcastic."
)

st.divider()


# --------------------------------------------------
# Input Section
# --------------------------------------------------

st.subheader("Enter a News Headline")

headline = st.text_input(
    "Headline",
    placeholder="Example: Government solves traffic problem by asking everyone to stay home"
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Sarcasm", use_container_width=True):

    if headline.strip() == "":
        st.warning("Please enter a news headline.")

    else:
        result = predict_sarcasm(headline)

        st.divider()
        st.subheader("Prediction Result")

        if result == "Sarcastic":
            st.error("🔴 Sarcastic")
        else:
            st.success("🟢 Non-Sarcastic")


# --------------------------------------------------
# Model Information
# --------------------------------------------------

st.divider()

st.subheader("Model Information")

col1, col2 = st.columns(2)

with col1:
    st.metric("Model", "Linear SVM")
    st.metric("Accuracy", "85.14%")

with col2:
    st.metric("Features", "TF-IDF")
    st.metric("F1-Score", "84.53%")


# --------------------------------------------------
# About the Project
# --------------------------------------------------

with st.expander("About this Project"):

    st.write(
        """
        This application detects sarcasm in news headlines using
        Natural Language Processing and Machine Learning.

        The text is preprocessed and converted into numerical
        TF-IDF features. A trained Linear Support Vector Machine
        (SVM) classifier then predicts whether the headline is
        sarcastic or non-sarcastic.
        """
    )

    st.write("**NLP Techniques:** Text preprocessing and TF-IDF")
    st.write("**Machine Learning Model:** Linear SVM")
    st.write("**Dataset:** News Headlines Dataset for Sarcasm Detection")