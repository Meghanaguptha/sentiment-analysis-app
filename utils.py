import numpy as np
import streamlit as st
import joblib
import os

# --- Project Root Directory ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Constants ---
MODEL_FILENAME = "sentiment_models.pkl"
VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"

@st.cache_resource
def load_model_and_vectorizer():
    """Load the sentiment model and vectorizer from disk."""
    try:
        # Try to find the model and vectorizer files in the current directory or parent directory
        model_path = None
        vectorizer_path = None
        
        # Check in current directory
        if os.path.isfile(os.path.join(BASE_DIR, MODEL_FILENAME)) and os.path.isfile(os.path.join(BASE_DIR, VECTORIZER_FILENAME)):
            model_path = os.path.join(BASE_DIR, MODEL_FILENAME)
            vectorizer_path = os.path.join(BASE_DIR, VECTORIZER_FILENAME)
        # Check in parent directory (for Streamlit Cloud deployment)
        elif os.path.isfile(os.path.join(os.path.dirname(BASE_DIR), MODEL_FILENAME)) and \
             os.path.isfile(os.path.join(os.path.dirname(BASE_DIR), VECTORIZER_FILENAME)):
            model_path = os.path.join(os.path.dirname(BASE_DIR), MODEL_FILENAME)
            vectorizer_path = os.path.join(os.path.dirname(BASE_DIR), VECTORIZER_FILENAME)
        else:
            st.error(f"Model or vectorizer files not found. Please make sure {MODEL_FILENAME} and {VECTORIZER_FILENAME} are in the project's root directory.")
            st.error(f"Current directory: {os.getcwd()}")
            st.error(f"Files in current directory: {os.listdir('.')}")
            return None, None
            
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
        
    except Exception as e:
        st.error(f"Error loading the sentiment analysis model: {str(e)}")
        return None, None

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    was_1d = False
    if x.ndim == 1:
        was_1d = True
        x = x.reshape(1, -1)
    max_x = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max_x)
    result = e_x / e_x.sum(axis=1, keepdims=True)
    return result.flatten() if was_1d else result