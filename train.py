import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def train_model(n_estimators):

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)

    # Cargar el conjunto de datos desde el archivo CSV (Dataset Wine)
    try:
        # Se asume un CSV con estructura similar: features numéricas y columna 'target'
        wine = pd.read_csv("data/wine_dataset.csv")
    except FileNotFoundError:
        print(
            "Error: El archivo 'data/wine_dataset.csv' no fue encontrado. Cargando desde sklearn para demostración."
        )
        # Fallback para asegurar que el código corra si no tienes el CSV
        wine_data = datasets.load_wine(as_frame=True)
        wine = wine_data.frame

    # Dividir el DataFrame en características (X) y etiquetas (y)
    X = wine.drop("target", axis=1)
    y = wine["target"]

    # Iniciar un experimento de MLflow
    with mlflow.start_run():
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Inicializar y entrenar el modelo (Gradient Boosting)
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        # Realizar predicciones y calcular la precisión
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        # Guardar el modelo entrenado en un archivo .pkl
        joblib.dump(model, "model.pkl")

        # Registrar el modelo con MLflow
        # mlflow.sklearn.log_model(model, "gradient-boosting-model")

        # Registrar parámetros y métricas
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_artifact("model.pkl")

        print(
            f"Modelo GradientBoosting entrenado con n_estimators = {n_estimators} y\
            precisión: {accuracy:.4f}"
        )
        print("Experimento registrado con MLflow.")

        # --- Sección de Reporte para CML ---
        # 1. Generar la matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Matriz de Confusión - Wine Dataset")
        plt.xlabel("Predicciones")
        plt.ylabel("Valores Reales")
        plt.savefig("confusion_matrix.png")
        print("Matriz de confusión guardada como 'confusion_matrix.png'")
        # --- Fin de la sección de Reporte ---

        mlflow.log_artifact("confusion_matrix.png")
        metrics = {"accuracy": accuracy}
        with open("mlflow_metrics.json", "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=150,
        help="Number of estimators for GradientBoostingClassifier",
    )
    args = parser.parse_args()
    train_model(args.n_estimators)
