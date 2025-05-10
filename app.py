import streamlit as st
import pickle

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# App Title
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection App")
st.write("Enter news content and click **Check** to see if it's FAKE or REAL.")

# Text input from user
user_input = st.text_area("Enter News Text:", height=200)

# When user clicks the "Check" button
if st.button("Check"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some news content.")
    else:
        # Transform and Predict
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        # Show result
        if prediction == "FAKE":
            st.error("🔴 This news is predicted to be **FAKE**.")
        else:
            st.success("🟢 This news is predicted to be **REAL**.")
