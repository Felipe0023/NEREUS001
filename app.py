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


import streamlit as st
from PIL import Image
logo = Image.open("LOGO_NEREUS.png")
st.image(logo, width=200)

# Configuración de página
st.set_page_config(page_title="Geología & Hidrogeología AI", layout="wide")

# --- FUNCIONES DE APOYO ---
def load_data(file):
    return pd.read_csv(file)

# --- SIDEBAR: CARGA DE ARCHIVOS ---
st.sidebar.header("1. Carga de Datos")
uploaded_csv = st.sidebar.file_uploader("Subir K_000_Datos.csv", type="csv")
uploaded_tif = st.sidebar.file_uploader("Subir K_000_Dem.tif", type="tif")

# Parámetros del modelo
st.sidebar.header("2. Configuración 3D")
paso_xy = st.sidebar.number_input("Paso XY (Lat/Lon)", value=0.005, format="%.3f")
paso_z = st.sidebar.number_input("Paso Z (Profundidad m)", value=20)
num_capas_z = st.sidebar.slider("Número de capas verticales", 1, 100, 41)
step_visual = st.sidebar.slider("Resolución visual Render", 1, 10, 5)

# --- CUERPO PRINCIPAL ---
st.title("🛰️ Sistema de Predicción de Conductividad Hidráulica")

if uploaded_csv and uploaded_tif:
    # Procesamiento inicial
    df_raw = load_data(uploaded_csv)

    tabs = st.tabs(["Análisis Exploratorio", "Entrenamiento", "Predicción 3D", "XAI (SHAP)"])

    # --- TAB 1: EDA ---
    with tabs[0]:

        st.subheader("Mapa de Localización")
        # Guardar temporalmente el TIF para rasterio
        with open("temp_dem.tif", "wb") as f:
            f.write(uploaded_tif.getbuffer())

        with rasterio.open("temp_dem.tif") as src:
            data_dem = src.read(1).astype(float)
            data_dem[data_dem == src.nodata] = np.nan
            bounds = src.bounds
            x_coords = np.linspace(bounds.left, bounds.right, data_dem.shape[1])
            y_coords = np.linspace(bounds.bottom, bounds.top, data_dem.shape[0])

        # Mapa Plotly
        fig_map = go.Figure()
        fig_map.add_trace(go.Heatmap(x=x_coords, y=y_coords, z=np.flipud(data_dem), colorscale='earth'))
        fig_map.add_trace(go.Scatter(x=df_raw['Longitud'], y=df_raw['Latitud'], mode='markers', marker=dict(color='red')))
        fig_map.update_layout(width=800, height=600, title="Ubicación de Perforaciones")
        st.plotly_chart(fig_map, use_container_width=True)


        st.subheader("Análisis de Datos de Campo")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Vista previa de datos:")
            st.dataframe(df_raw.head())

        with col2:
            # Comparación Litológica (Tu código original de Boxplot)
            st.write("Distribución de Conductividad por Roca")
            fig_box, ax_box = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df_raw, x='K', y='Tipo_Roca', palette="viridis", ax=ax_box)
            ax_box.set_xscale('log')
            st.pyplot(fig_box)

    # --- TAB 2: ENTRENAMIENTO ---
    with tabs[1]:
        st.subheader("Entrenamiento del Modelo XGBoost")
        if st.button("Ejecutar Entrenamiento"):
            with st.spinner("Entrenando modelos..."):
                # Aquí encapsulas tu lógica de:
                # 1. DBSCAN (HGS)
                # 2. Preprocesamiento (StandardScaler)
                # 3. XGBRegressor

                # Ejemplo de métricas:
                st.success("Modelo entrenado con éxito")
                st.metric("R² Score", "0.8452", delta="0.02")

                # Matriz de confusión (Si entrenas el clasificador)
                st.write("Matriz de Confusión - Tipos de Roca")
                # (Generar y mostrar plot)

    # --- TAB 3: PREDICCIÓN 3D ---
    with tabs[2]:
        st.subheader("Visualización del Subsuelo")

        # Lógica para generar la malla de puntos (Sección 08 de tu código)
        if st.button("Generar Modelo 3D"):
            # Nota: rasterio necesita leer el archivo TIF subido
            with rasterio.open(uploaded_tif) as src:
                # ... tu lógica de generación de puntos ...
                pass

            # Gráfico Plotly
            fig_3d = go.Figure()
            # Agregar superficie y puntos...
            st.plotly_chart(fig_3d, use_container_width=True)

    # --- TAB 4: EXPLICABILIDAD ---
    with tabs[3]:

        st.subheader("Interpretación del Modelo (SHAP)")
        st.info("Esta sección explica qué variables afectan más a la permeabilidad (K).")
        # Aquí llamarías a shap.summary_plot
        # Para Streamlit usa: st.pyplot(plt.gcf()) después de generar el plot de SHAP

else:
    st.warning("👈 Por favor, sube los archivos CSV y TIF en la barra lateral para comenzar.")
    st.image("https://drive.google.com/uc?export=view&id=1jcdYjLgdkbgomF81QpfbGQYDwePbgIkj", caption="Mi imagen PNG transparente")



