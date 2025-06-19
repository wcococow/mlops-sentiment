# Training script
import joblib
import mlflow
import mlflow.sklearn
from src.utils import prepare_datasets
from src.model import build_model

def train(path_to_data, model_output):
    mlflow.set_experiment("sentiment-classification")
    with mlflow.start_run():
        X_train, X_test, y_train, y_test, vectorizer = prepare_datasets(path_to_data)
        model = build_model()
        model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        joblib.dump(model, model_output)
        joblib.dump(vectorizer, "vectorizer.joblib")
        print("Model trained and logged to MLflow.")
