from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import numpy as np


# Carga el modelo
model = load_model("modelo/model_a.h5")

def predict_image(img):
    # Preprocesar la imagen para que coincida con las dimensiones de entrada del modelo
    img = img.resize((250, 250))  # Ajusta el tamaño según la configuración del modelo
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Añade una dimensión extra para representar el lote
    # Realizar predicciones con el modelo cargado
    predictions = model.predict(img_array)
    indice_maximo = np.argmax(predictions)
    # Encuentra el valor máximo
    valor_maximo = predictions[0, indice_maximo]
    print("El valor máximo es:", valor_maximo)
    print("El índice del valor máximo es:", indice_maximo)
    
    return indice_maximo
