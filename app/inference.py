import os
import joblib
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Paths to model artifacts
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "models/vectorizer.joblib")

# Load model and vectorizer
def load_model_and_vectorizer():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Model or vectorizer not found at specified paths.")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

# Predict sentiment
def predict_sentiment(text, model=None, vectorizer=None):
    if model is None or vectorizer is None:
        model, vectorizer = load_model_and_vectorizer()

    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    return prediction
