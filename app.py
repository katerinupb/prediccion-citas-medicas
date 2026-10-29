# -*- coding: utf-8 -*-
"""Prototipo académico de predicción de citas médicas."""

import base64
import pickle
import pandas as pd
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

# Recuperar el modelo entrenado. Si hay un problema de compatibilidad,
# permitir visualizar el formulario e informar del error.
modeloGB = None
labelencoder = None
variables = None
error_modelo = None
try:
    with open(carpeta / "modelo_citas_medicas.pkl", "rb") as archivo:
        modeloGB, labelencoder, variables = pickle.load(archivo)
except Exception as error:
    error_modelo = f"{type(error).__name__}: {error}"

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

# Sección 1. Información del paciente
st.subheader("Información del paciente")

col1, col2 = st.columns(2)

with col1:
    sexopaciente = st.selectbox("Sexo del paciente", ["F", "M"])

    grupo_edad = st.selectbox("Grupo de edad", ["ADOLESCENCIA", "ADULTEZ", "INFANCIA", "JUVENTUD", "PRIMERA_INFANCIA", "VEJEZ"])

    programacronico = st.selectbox( "Programa de atención de enfermedades crónicas",  ["Diabetes", "Hipertension", "Nefroproteccion", "Ninguno", "Sindrome Metabolico"])

with col2:
    poblacion_vulnerable = st.selectbox(
        "Población vulnerable",
        ["Adulta Mayor sin Protección", "Afrodescendiente",
         "Desplazada", "Indigena", "LGTBI", "Ninguna",
         "Poblacion Trabajo Sexual", "Poblacion a Cargo ICBF",
         "Situación de Calle", "Victima Conflicto Armado"]
    )

    grupo_municipio = st.selectbox("Tiempo estimado de desplazamiento a la sede", ["0 a 1 horas", "1 a 2 horas", "2 a 4 horas","4 a 6 horas", "mas de 6 horas"])

# Sección 2. Información de la cita
st.subheader("Información de la cita")

col1, col2 = st.columns(2)

with col1:
    funcionalidad = st.selectbox("Tipo de atención", ["CONSULTA", "LABORATORIO", "PROCEDIMIENTO", "TERAPIAS", "TRATAMIENTO ODONTOLOGICO"])

    grupo_codigocups = st.selectbox(
        "Grupo de procedimiento o servicio",
        ["CONSULTAS CLINICAS Y OTROS PROFESIONALES",
         "LABORATORIO CLINICO",
         "ODONTOLOGIA ESPECIALIZADA",
         "ODONTOLOGIA GENERAL Y PREVENTIVA",
         "PATOLOGIA",
         "PROCEDIMIENTOS DE ENFERMERIA",
         "TERAPIAS Y REHABILITACION"]
    )

    grupo_pym = st.selectbox("Tipo de programa", ["CITA MORBILIDAD", "PROMOCION, PREVENCION Y PROGRAMAS ESPECIALES"])

with col2:
    grupo_convenionombre = st.selectbox( "Convenio o entidad responsable", ["Coosalud", "Nueva EPS Contributivo",
         "Nueva EPS Subsidiado", "Otros", "Particular",
         "Proteger EPS Contributivo", "Proteger EPS Subsidiado"]
    )

    municipiosede = st.selectbox( "Municipio de la sede", ["BUENAVISTA", "CAIMITO", "COROZAL", "GUARANDA",
         "MAJAGUAL", "SAMPUES", "SAN ANTONIO DE PALMITO",
         "SAN BENITO", "SAN LUIS DE SINCE", "SAN MARCOS",
         "SAN PEDRO", "SANTIAGO DE TOLU", "SINCELEJO", "SUCRE"]
    )

    rotulo = st.selectbox("Rótulo del servicio",
        ["AGENDAMIENTO LABORATORIO CLINICO",
         "AGENDAMIENTO PATOLOGIA", "CAMBIO DE SONDA",
         "CONSULTA PRIMERA VEZ ODONTOLOGIA",
         "CONSULTA TRATAMIENTO ODONTOLOGIA",
         "MONITOREO FETAL", "MONITOREO PRESION ARTERIAL",
         "PROCEDIMIENTOS COLPOSCOPIA",
         "PROCEDIMIENTOS DE ELECTROCARDIOGRAMA",
         "PROCEDIMIENTOS MENORES ENFERMERIA",
         "SIN ROTULO", "TOMA DE MUESTRAS DOMICILIARIAS"]
    )

# Sección 3. Programación de la cita
st.subheader("Programación de la cita")

col1, col2 = st.columns(2)

with col1:
    oportunidad = st.number_input(
        "Oportunidad de la cita (días)",
        min_value=0,
        max_value=150,
        value=0,
        step=1
    )

    dia_semana = st.selectbox("Día de la semana",["DOMINGO", "JUEVES", "LUNES", "MARTES","MIERCOLES", "SABADO", "VIERNES"])

    mes = st.selectbox("Mes de la cita", ["AGOSTO", "JULIO", "SEPTIEMBRE"])

with col2:
    grupo_horacita = st.selectbox("Horario de la cita", ["MADRUGADA", "MANIANA", "MEDIODIA", "NOCHE", "TARDE"])

    turno_doble = st.selectbox("Turno doble",["NO", "SI"])

"""### Preparar los datos para la predicción"""

# Preparar los datos ingresados en el formulario
datos = pd.DataFrame([{
    'oportunidad': oportunidad,
    'sexopaciente': sexopaciente,
    'programacronico': programacronico,
    'funcionalidad': funcionalidad,
    'rotulo': rotulo,
    'municipiosede': municipiosede,
    'turno_doble': turno_doble,
    'poblacion_vulnerable': poblacion_vulnerable,
    'grupo_convenionombre': grupo_convenionombre,
    'grupo_municipio': grupo_municipio,
    'dia_semana': dia_semana,
    'mes': mes,
    'grupo_horacita': grupo_horacita,
    'grupo_edad': grupo_edad,
    'grupo_pym': grupo_pym,
    'grupo_codigocups': grupo_codigocups
}])

# Mantener el orden de las variables del modelo
# Utilizar el orden exacto del entrenamiento cuando el modelo esté disponible
if variables is not None:
    datos = datos[list(variables)]

"""### Ejecutar la predicción"""

# Ejecutar la predicción cuando el usuario presione el botón
if error_modelo:
    st.error(f"No se pudo cargar el modelo: {error_modelo}")

if st.button("🔍 Predecir estado de la cita", disabled=modeloGB is None):

    # Obtener la predicción del modelo
    prediccion = modeloGB.predict(datos)

    # Convertir el resultado numérico a su categoría original
    estado_predicho = labelencoder.inverse_transform(prediccion)[0]

    # Mostrar el resultado
    st.success(f"Estado predicho de la cita: {estado_predicho}")

# Informar sobre el desempeño del modelo
st.warning(
    "El modelo obtuvo una exactitud del 58,73 % en los datos de prueba. "
    "Las predicciones son orientativas y deben interpretarse con precaución."
)