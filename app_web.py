import json
import os

import requests
import streamlit as st

# Configuración de página
st.set_page_config(page_title="Wine Prediction", layout="wide")
st.title("API de Predicción: Dataset Wine")

# Lista ordenada de características según el estándar de sklearn/Wine dataset
feature_names = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines",
    "proline",
]

input_data = {}
col1, col2 = st.columns(2)

with col1:
    input_data["alcohol"] = st.slider("Alcohol", 11.0, 15.0, 13.0)
    input_data["malic_acid"] = st.slider("Malic Acid", 0.74, 5.80, 2.34)
    input_data["ash"] = st.slider("Ash", 1.36, 3.23, 2.36)
    input_data["alcalinity_of_ash"] = st.slider("Alcalinity of Ash", 10.6, 30.0, 19.5)
    input_data["magnesium"] = st.slider("Magnesium", 70.0, 162.0, 99.7)
    input_data["total_phenols"] = st.slider("Total Phenols", 0.98, 3.88, 2.29)
    input_data["flavanoids"] = st.slider("Flavanoids", 0.34, 5.08, 2.03)

with col2:
    input_data["nonflavanoid_phenols"] = st.slider(
        "Nonflavanoid Phenols", 0.13, 0.66, 0.36
    )
    input_data["proanthocyanins"] = st.slider("Proanthocyanins", 0.41, 3.58, 1.59)
    input_data["color_intensity"] = st.slider("Color Intensity", 1.3, 13.0, 5.1)
    input_data["hue"] = st.slider("Hue", 0.48, 1.71, 0.96)
    input_data["od280/od315_of_diluted_wines"] = st.slider(
        "OD280/OD315", 1.27, 4.00, 2.61
    )
    input_data["proline"] = st.slider("Proline", 278.0, 1680.0, 746.0)

if st.button("Obtener Predicción"):
    # Construir el vector de características respetando el orden exacto
    features = [input_data[name] for name in feature_names]

    # La API espera una lista de features
    payload = {"features": features}

    # URL del servicio Flask - Fixed default to use container name
    api_url = os.environ.get("API_URL", "http://mlops-fp-api:5000/predict")

    try:
        response = requests.post(
            api_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10,  # Add timeout
        )

        if response.status_code == 200:
            prediction_result = response.json().get("prediction")
            # Nombres reales de los tipos de vino del dataset Wine
            species_map = {
                0: "Barolo",
                1: "Grignolino",
                2: "Barbera",
            }
            predicted_class = species_map.get(
                prediction_result, f"Clase {prediction_result}"
            )
            st.success(f"Predicción del modelo: {predicted_class}")
        else:
            st.error(f"Error del servidor: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error de conexión: {e}")
        st.info("Asegúrate de que el servicio API esté ejecutándose correctamente.")
