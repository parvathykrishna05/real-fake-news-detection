import streamlit as st
import joblib
import requests
import io
import re
import string

# --- GitHub Release URLs (update these if you change version or user/repo) ---
model_url = 'https://objects.githubusercontent.com/github-production-release-asset-2e65be/999677485/490f0a96-bb07-47cc-b77d-338fb84e98e8?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=releaseassetproduction%2F20250610%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250610T164549Z&X-Amz-Expires=300&X-Amz-Signature=69d1ee2e4ccee9733bc011aeef88fe30350d67b23e813f9dd6f62ff5f300ee1a&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3Dmodel.pkl&response-content-type=application%2Foctet-stream'
tfidf_url = 'https://objects.githubusercontent.com/github-production-release-asset-2e65be/999677485/0114bc17-bab5-4ccc-b739-18d78b0a2491?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=releaseassetproduction%2F20250610%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250610T164719Z&X-Amz-Expires=300&X-Amz-Signature=1dfb9fa015ae02b73b194eeb8a844da320d920e8064e9107b4b44b1942782370&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3Dtfidf_compressed.pkl&response-content-type=application%2Foctet-stream'

# --- Load model and vectorizer from GitHub Release ---
@st.cache_resource
def load_model():
    model_response = requests.get(model_url)
    return joblib.load(io.BytesIO(model_response.content))

@st.cache_resource
def load_vectorizer():
    tfidf_response = requests.get(tfidf_url)
    return joblib.load(io.BytesIO(tfidf_response.content))

model = load_model()
tfidf = load_vectorizer()

# --- Text Cleaning Function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# --- Streamlit UI ---
st.title("📰 Fake News Detection App")
st.markdown("Predict whether the given news text is **Real or Fake**, using a machine learning model trained on real-world data.")

input_text = st.text_area("✍️ Enter news content below:")

if st.button("Check"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some news content.")
    else:
        cleaned = clean_text(input_text)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)

        if prediction[0] == 1:
            st.success("✅ This looks like *Real News*.")
        else:
            st.error("🚫 This appears to be *Fake News*.")
