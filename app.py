import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from PIL import Image
from io import BytesIO
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from xgboost import XGBRegressor, XGBClassifier
import requests
import pydeck as pdk
import os
import sys

import appk001ubicacion
import appk002zonaestud
import appk003preproces
import appk004modelalit
import appk005pronoslit
import appk006modlog10k
import appk007prolog10k
import appk008graficars

mapbox_key = st.secrets.get("MAPBOX_TOKEN", "")

# Configuración de la página (Modo ancho para mejor visualización)
st.set_page_config(page_title="Gemelos Digitales - Acuíferos", layout="wide")

# ==========================================================================================================================================
# MECANISMO DE CACHÉ: INMUTABILIDAD DE IMÁGENES EN RAM 
# ==========================================================================================================================================
@st.cache_data(show_spinner=False)
def cargar_logo_fijo(ruta_archivo):
    try:
        return Image.open(ruta_archivo)
    except FileNotFoundError:
        return None

# Precargamos los logos en la caché persistente global
logo_nereus_cache = cargar_logo_fijo("logo_nereus.png")
logo_nuevo_cache = cargar_logo_fijo("logo_licenciado.png") or cargar_logo_fijo("logo_nuevo.jpg")

# ==========================================================================================================================================
# GESTIÓN DE MEMORIA RAM (Session State)
# ==========================================================================================================================================
if "K001_datos" not in st.session_state:
    st.session_state["K001_datos"] = None
if "K001_dem" not in st.session_state:
    st.session_state["K001_dem"] = None
if "procesar_click" not in st.session_state:
    st.session_state["procesar_click"] = False

# ==========================================================================================================================================
# 1. BARRA LATERAL (SIDEBAR) - ENCERRADA EN FORMULARIO PARA EVITAR REFRESCADO INTERMEDIO
# ==========================================================================================================================================
with st.sidebar.form("formulario_carga"):
    archivo_csv = st.file_uploader("Cargar archivo CSV", type=["csv"])
    archivo_tif = st.file_uploader("Cargar archivo TIF (DEM)", type=["tif", "tiff"])
    st.markdown("---")
    # En un formulario, el botón debe ser obligatoriamente un form_submit_button
    boton_procesar = st.form_submit_button("PROCESAR DATOS")

# ==========================================================================================================================================
# 2. GESTIÓN DE ENTRADA DE DATOS (SOLO SE EJECUTA AL HACER CLICK EN EL BOTÓN)
# ==========================================================================================================================================
if boton_procesar:
    st.session_state["procesar_click"] = True
    if archivo_csv is not None:
        try:
            st.session_state["K001_datos"] = pd.read_csv(archivo_csv)
        except Exception as e:
            st.sidebar.error(f"Error al leer CSV: {e}")
    else:
        st.session_state["K001_datos"] = None
    if archivo_tif is not None:
        try:
            st.session_state["K001_dem"] = archivo_tif.read()
        except Exception as e:
            st.sidebar.error(f"Error al leer TIF: {e}")
    else:
        st.session_state["K001_dem"] = None

# Estado de validación riguroso
ambos_archivos_listos = st.session_state["K001_datos"] is not None and st.session_state["K001_dem"] is not None

# Si el usuario cambia o remueve archivos, bajamos la bandera para evitar inconsistencias
if not ambos_archivos_listos:
    st.session_state["procesar_click"] = False

# ==========================================================================================================================================
# CREACIÓN DE ALIAS CORTOS DESDE EL SESSION STATE (EXTRACCIÓN SEGURA)
# ==========================================================================================================================================
K001_dem = st.session_state["K001_dem"]
K001_datos = st.session_state["K001_datos"]

# ==========================================================================================================================================
# 3. CUERPO CENTRAL DE LA PÁGINA - CON CONTENEDOR AISLADO (FRAGMENTO)
# ==========================================================================================================================================
@st.fragment
def renderizar_cabecera_estatica(click_procesar, archivos_listos):
    col1, col2, col3 = st.columns([1, 1.1, 1])
    with col2:
        contenedor_logo = st.empty()
        
        # Evaluación atómica sin disparar cargas de disco repetitivas
        if click_procesar and archivos_listos:
            if logo_nuevo_cache is not None:
                contenedor_logo.image(logo_nuevo_cache, use_container_width=True)
            else:
                contenedor_logo.markdown(
                    "<div style='text-align:center; padding:40px; background:#f0f2f6; border-radius:10px; color:#333; font-weight:bold;'>"
                    "✨ [ LOGO NUEVO - Coloque 'logo_licenciado.png' en su carpeta ] ✨"
                    "</div>", 
                    unsafe_allow_html=True
                )
        else:
            if logo_nereus_cache is not None:
                contenedor_logo.image(logo_nereus_cache, use_container_width=True)
            else:
                contenedor_logo.error("⚠️ Archivo 'logo_nereus.png' no encontrado.")

# Invocamos la cabecera protegida pasándole el estado actual de las variables
renderizar_cabecera_estatica(st.session_state["procesar_click"], ambos_archivos_listos)


# ==========================================================================================================================================
# 4. CONTROL DE FLUJO Y SECCIONES (TABS)
# ==========================================================================================================================================
if not ambos_archivos_listos:
    if st.session_state["procesar_click"]:
        st.write(""); st.write("")
    else:
        st.write(""); st.write("") 
else:
    if not st.session_state["procesar_click"]:
        st.markdown("")
    else:
        st.markdown("<h4 style='text-align: center;'>Gemelos Digital - Gestión de Acuíferos</h4>", unsafe_allow_html=True)

        # Renderizado de los Múltiples Tabs de la Plataforma
        tabs = st.tabs([
            "Ubicación", "Zona de Pronóstico", "Preprocesamiento", 
            "Modelamiento Litológico", "Pronóstico Litológico",  
            "Modelamiento de Log10 K", "Pronóstico Log10 K",  
            "Gráfico 3D"
        ])
        with tabs[0]: # Ubicación
            appk001ubicacion.BLOQUE001(K001_datos, K001_dem)
            appk001ubicacion.BLOQUE002(K001_datos, mapbox_key) 
            appk001ubicacion.BLOQUE003(K001_datos, K001_dem, submuestreo=5)
        with tabs[1]: # Zona de Pronóstico
            appk002zonaestud.BLOQUE001(K001_dem) # K001_datos_Nuevos.csv/K_008_Dem.tif a K_008_Datos_Nuevos.csv
        with tabs[2]: # Preprocesamiento
            appk003preproces.BLOQUE001() # K001_Datos_Nuevos.csv a K002_datos_limpio.csv
            appk003preproces.BLOQUE002() # K002_datos_limpio.csv a K003_reescalado.csv K003_scaler_ml.joblib 
            appk003preproces.BLOQUE003() # K003_reescalado.csv a K004_hgs.csv
        with tabs[3]: # Modelamiento Litológico
            appk004modelalit.BLOQUE001() # K004_hgs.csv a 
        with tabs[4]: # Pronóstico Litológico  
            appk005pronoslit.BLOQUE001() # K003_scaler_ml.joblib K008_Datos_Nuevos_Pronostico.csv obt. K009_reescalado.csv
            appk005pronoslit.BLOQUE002() # K009_reescalado.csv obt. K010_hgs.csv
            appk005pronoslit.BLOQUE003() # K010_hgs.csv a K010_hgs.csv a K011_Nuevos_Datos_Pronostico.csv
        with tabs[5]: # Modelamiento de Log10 K    
            appk006modlog10k.BLOQUE001() # K004_hgs.csv
        with tabs[6]: # Pronóstico Log10 K
            appk007prolog10k.BLOQUE001() # K003_scaler_ml K011_Nuevos_Datos_Pronostico a K012_Escaleado.csv
            appk007prolog10k.BLOQUE002() # K012_Escaleado.csv a K013_hgs.csv
            appk007prolog10k.BLOQUE003() # K013_hgs.csv a K014_resultado.csv
        #with tabs[7]: # Graficar
        #   appk008graficars.BLOQUE001(K014_resultado)
        #*******************************************************************************************************




