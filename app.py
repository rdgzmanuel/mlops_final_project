import joblib
import numpy as np
from flask import Flask, Response, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

# Contador de Prometheus para predicciones por tipo de vino
PREDICTION_COUNTER = Counter(
    "wine_prediction_count",
    "Contador de predicciones del modelo Wine por tipo de vino",
    ["wine_type"],
)

# Cargar el modelo entrenado
try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    print(
        "Error: 'model.pkl' no encontrado. Por favor, asegúrate de haber ejecutado el"
        "script de entrenamiento."
    )
    model = None

# Inicializar la aplicación Flask
app = Flask(__name__)


@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "Modelo no cargado. Por favor, entrene el modelo primero."
        }), 500

    try:
        # Obtener los datos de la petición en formato JSON
        data = request.get_json(force=True)
        features = np.array(data["features"]).reshape(1, -1)

        # Realizar la predicción
        prediction = model.predict(features)
        prediction_int = int(prediction[0])

        # Mapear el resultado numérico a un tipo de vino
        wine_map = {0: "Barolo", 1: "Grignolino", 2: "Barbera"}
        predicted_wine = wine_map.get(prediction_int, "unknown")

        # Incrementa el contador para el tipo de vino predicho
        PREDICTION_COUNTER.labels(wine_type=predicted_wine).inc()

        return jsonify({"prediction": prediction_int, "wine_type": predicted_wine})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    print("Iniciando API en puerto 5000...")
    app.run(host="0.0.0.0", port=5000)
