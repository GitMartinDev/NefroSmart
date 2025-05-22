import pickle
import numpy as np
import os

# Ruta al modelo
modelo_path = os.path.join(os.path.dirname(__file__), "modelo_entrenado.pkl")

# Cargar el modelo
with open(modelo_path, "rb") as f:
    modelo = pickle.load(f)

# Función para predecir
def predecir(datos):
    try:
        if not isinstance(datos, list) or len(datos) != 24:
            raise ValueError(f"Se esperaban exactamente 24 valores numéricos, pero se recibieron {len(datos)}.")

        datos_array = np.array(datos).reshape(1, -1)
        prediccion = modelo.predict(datos_array)[0]
        return int(prediccion)
    except Exception as e:
        raise RuntimeError(f"Error en la predicción: {e}")
