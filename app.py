# -*- coding: utf-8 -*-
"""Prototipo académico de predicción de citas médicas."""

import base64
import pickle
from pathlib import Path

import streamlit as st
from PIL import Image

# Componentes utilizados al entrenar el modelo
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC


# Estas funciones deben existir para recuperar el pipeline guardado con pickle.
def estrategia_undersampling(Y_train):
    cantidad_activos = (Y_train == 0).sum()
    return {0: int(cantidad_activos * 0.5)}


def estrategia_balanceo(Y_train):
    cantidad_canceladas = (Y_train == 1).sum()
    return {1: cantidad_canceladas * 2}


# Rutas relativas a app.py
carpeta = Path(__file__).resolve().parent

# Configuración de la página
icono = Image.open(carpeta / "images" / "icono_corazon.png")
st.set_page_config(
    page_title="Predicción de Citas Médicas",
    page_icon=icono,
    layout="wide",
)

# Recuperar el modelo entrenado
with open(carpeta / "modelo_citas_medicas.pkl", "rb") as archivo:
    modeloGB, labelencoder, variables = pickle.load(archivo)

# Cargar CSS e insertar la imagen de fondo para que funcione en Streamlit Cloud
estilos = (carpeta / "estilos.css").read_text(encoding="utf-8")
imagen_fondo = (carpeta / "images" / "fondo_medico.png").read_bytes()
fondo_base64 = base64.b64encode(imagen_fondo).decode("utf-8")
estilos = estilos.replace(
    'url("images/fondo_medico.png")',
    f'url("data:image/png;base64,{fondo_base64}")',
)
st.markdown(f"<style>{estilos}</style>", unsafe_allow_html=True)

# Encabezado
st.markdown(
    """
    <div class="encabezado">
        <h1>Predicción de Citas Médicas</h1>
        <p>Prototipo demostrativo | IPS Salud a Tu Lado</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Herramienta académica en desarrollo. Las predicciones tienen limitaciones "
    "de precisión y no deben utilizarse como único criterio para decisiones operativas."
)
