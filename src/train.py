import argparse
import joblib
import mlflow
import mlflow.sklearn
from src.utils import prepare_datasets
from src.model import build_model
import os


def train(path_to_data, model_output):
    # Set experiment name
    mlflow.set_experiment("sentiment-classification")

    # Load and preprocess data
    print(f"Loading data from {path_to_data}...")
    X_train, X_test, y_train, y_test, vectorizer = prepare_datasets(path_to_data)

    # Build and train model
    model = build_model()
    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    acc = model.score(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}")

    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_param("model_type", model.__class__.__name__)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

    # Save model and vectorizer
    output_dir = os.path.dirname(model_output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, model_output)
    joblib.dump(vectorizer, "vectorizer.joblib")

    print(" Model trained, saved, and logged to MLflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sentiment classification model.")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file with 'text' and 'label'")
    parser.add_argument("--output", type=str, default="model.joblib", help="Output path to save the model")

    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f" Data file not found at: {args.data}")

    train(args.data, args.output)
