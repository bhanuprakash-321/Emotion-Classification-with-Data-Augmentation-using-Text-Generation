import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import re

# Set the title
st.title("Emotion Detection from Text")

# Load the model
try:
    model = tf.keras.models.load_model("emotion_model.h5")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Load the tokenizer
try:
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
except FileNotFoundError:
    st.error("Tokenizer file 'tokenizer.pkl' not found. Please ensure it's in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the tokenizer: {e}")
    st.stop()

# Load the label encoder
try:
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
except FileNotFoundError:
    st.error("Label encoder file 'label_encoder.pkl' not found. Please ensure it's in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the label encoder: {e}")
    st.stop()

# Emotion to Emoji Mapping
emotion_emojis = {
    "joy": "😂",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😮"
}

max_length = 60  # Ensure this matches the training max sequence length

# Input Section
user_input = st.text_input("Enter a sentence to detect emotion:")

# Prediction Logic
if user_input:
    cleaned_input = re.sub(r"[^a-zA-Z\s]", "", user_input.lower()).strip()
    input_seq = tokenizer.texts_to_sequences([cleaned_input])
    padded_input = pad_sequences(input_seq, maxlen=max_length, padding='post', truncating='post')

    # Make prediction
    prediction = model.predict(padded_input)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_emotion = le.inverse_transform(predicted_class)[0]
    emoji = emotion_emojis.get(predicted_emotion, "")

    # Display Result
    st.subheader("Predicted Emotion:")
    st.markdown(f"<h2 style='text-align: center;'>{predicted_emotion} {emoji}</h2>", unsafe_allow_html=True)
