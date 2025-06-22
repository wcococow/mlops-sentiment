from fastapi import FastAPI
from pydantic import BaseModel
from app.inference import load_model_and_vectorizer, predict_sentiment

# Load model/vectorizer once at startup
model, vectorizer = load_model_and_vectorizer()

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input_data: TextInput):
    sentiment = predict_sentiment(input_data.text, model, vectorizer)
    return {"text": input_data.text, "sentiment": sentiment}
