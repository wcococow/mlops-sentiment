# FastAPI serving script
from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(input: TextInput):
    features = vectorizer.transform([input.text])
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}

