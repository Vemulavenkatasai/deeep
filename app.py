import streamlit as st
import numpy as np
import pickle
from gensim.models import Word2Vec

# -----------------------------
# Load pretrained objects
# -----------------------------
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        clf = pickle.load(f)

    w2v_model = Word2Vec.load("word2vec.model")
    return clf, w2v_model

model, w2v_model = load_artifacts()

# -----------------------------
# Helper: text → Word2Vec vector
# -----------------------------
def text_to_vector(text, w2v_model):
    words = text.lower().split()
    vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]

    if len(vectors) == 0:
        return np.zeros(w2v_model.vector_size)

    return np.mean(vectors, axis=0)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Mental Health Text Classifier", layout="centered")

st.title("🧠 Mental Health Text Classifier (Word2Vec)")
st.write("Enter text to detect mental health risk.")

user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        vec = text_to_vector(user_input, w2v_model)
        vec = vec.reshape(1, -1)

        prob = model.predict_proba(vec)[0][1]
        label = "HIGH RISK" if prob >= 0.5 else "LOW RISK"

        st.subheader("Prediction Result")
        st.write(f"**Risk Level:** {label}")
        st.write(f"**Confidence Score:** {prob:.3f}")
