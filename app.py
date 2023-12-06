from PIL import Image
import streamlit as st
import pandas as pd
from model import predict_image
from PIL import Image

class App:
    def __init__(self):
        self.titulo = "Predicción del cáncer de piel"
        

    def run(self):
        st.set_page_config(page_title=self.titulo, layout="wide")
        # Titulo de la aplicación
        st.title(self.titulo)

        st.title("Bienvenido")
        # pasos
        st.subheader("Pasos")
        st.markdown("""
        1. Cargar una imagen de la muestra.
        2. Observar la prediccion debajo de la imagen.
        """)
        # Cargar imagen
        st.subheader("Cargar imagen")
        uploaded_file = st.file_uploader("Elige la muestra de cancer...", type="jpg")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Imagen cargada.', use_column_width=True)
            st.write("")
            st.write("Clasificando...")
            label = predict_image(image)
            st.write(label)
            st.write("")
            



# Ejecutar la aplicación
if __name__ == "__main__":
    app = App()
    app.run()