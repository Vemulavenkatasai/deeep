import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import pickle

# -----------------------------
# Load pretrained objects
# -----------------------------
# 1️⃣ LSTM model
model = load_model("lstm_model.keras")  # your trained model

# 2️⃣ Tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# 3️⃣ Max sequence length used during training
MAX_LEN = 100  # replace with your training sequence length

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Mental Health Text Classifier (Word2Vec + LSTM)")

user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # -----------------------------
        # Preprocess input
        # -----------------------------
        seq = tokenizer.texts_to_sequences([user_input])
        padded_seq = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

        # -----------------------------
        # Make prediction
        # -----------------------------
        prob = model.predict(padded_seq)[0][0]  # assuming output sigmoid
        label = 1 if prob >= 0.5 else 0
        risk = "HIGH RISK" if label == 1 else "LOW RISK"

        # -----------------------------
        # Display results
        # -----------------------------
        st.write(f"**Predicted Label:** {label} ({risk})")
        st.write(f"**Confidence:** {prob:.3f}")
