import streamlit as st
import pickle
from utils import clean_text

# Load model and vectorizer
model = pickle.load(open('model/model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

# App title
st.title("📩 Spam Detection App")

st.write("Enter a message below to check if it's Spam or Not")

# Input box
message = st.text_area("Enter your message:")

# Predict button
if st.button("Predict"):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        # Clean text
        cleaned = clean_text(message)
        
        # Convert to vector
        vector = vectorizer.transform([cleaned])
        
        # Predict
        prediction = model.predict(vector)[0]
        
        # Output
        if prediction == 1:
            st.error("🚨 This is a SPAM message")
        else:
            st.success("✅ This is NOT spam")