import pandas as pd
from sklearn.datasets import load_wine

# Cargar dataset
wine = load_wine()

# Crear DataFrame
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
df["target"] = wine.target

# Guardar a CSV
df.to_csv("data/wine_dataset.csv", index=False)
print("Archivo wine_dataset.csv creado exitosamente.")
