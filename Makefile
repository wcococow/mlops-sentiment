# MLOps Sentiment Makefile with DVC and MLflow

# Variables
DATA=data/imdb.csv
MODEL=models/sentiment_model.joblib
VECTORIZER=vectorizer.joblib

# Run full pipeline using DVC
pipeline:
	dvc repro

# Stage to run training via DVC
train:
	python -m src.train --data $(DATA) --output $(MODEL)

# Stage to run evaluation
eval:
	python -m src.eval --model $(MODEL) --vectorizer $(VECTORIZER) --data data/imdb_test.csv

# Serve API locally with FastAPI
serve:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Re-run all (train + eval)
all: train eval

# MLflow UI
mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

# Docker
docker-build:
	docker build -t sentiment-api .

docker-run:
	docker run -p 8000:8000 sentiment-api

# Clean model artifacts (optional)
clean:
	rm -f $(MODEL) $(VECTORIZER)

.PHONY: train eval serve all pipeline mlflow-ui docker-build docker-run clean
