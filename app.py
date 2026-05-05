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
import requests

#***************************************************************************
logo = Image.open("LOGO_NEREUS.png")
st.set_page_config(page_title="NEREUS V.1", page_icon=logo)
# 2. Crear tres columnas
# La proporción [1, 1, 1] crea tres espacios iguales. 
# Pondremos el logo en la columna del medio.
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(logo, width=300)
#***************************************************************************



#***************************************************************************
# Configuración de página

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
#***************************************************************************

# --- CUERPO PRINCIPAL ---
st.title("Digital Twin para la Gestión de Acuíferos")

if uploaded_csv and uploaded_tif:
    # Procesamiento inicial
    df_raw = load_data(uploaded_csv)

    tabs = st.tabs(["Análisis Exploratorio", "Entrenamiento", "Predicción 3D", "XAI (SHAP)"])
    #****************************************************************************************
    # --- TAB 1: EDA ---
    with tabs[0]:
        #************************************************************************************
        st.header("Mapa de Localización")
        # Guardar temporalmente el TIF para rasterio
        with st.container(border=True):
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
            #fig_map.update_layout(width=800, height=600, title="Ubicación de Perforaciones")
            fig_map.update_layout(width=800, height=600)
            st.plotly_chart(fig_map, use_container_width=True)

        #****************************************************************************************
        with st.container(border=True):
            st.header("Monitoreo de Perforaciones")
            # Fila de Métricas
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Perforaciones", len(df_raw))
            with c2:
                st.metric("Promedio Valor K", f"{df_raw['K'].mean():.2f}")
            with c3:
                st.metric("Profundidad Promedio", f"{df_raw['Profundidad'].mean():.1f} m")

        #****************************************************************************************       
        with st.container(border=True):
            #st.subheader("Análisis de Datos de Campo")
            # 1. Función de color basada en tu columna 'K'
            def color_por_k(k_val):
                try:
                    # Si K es muy pequeño o negativo (log), usamos abs para la intensidad
                    v = float(k_val)
                    intensidad = min(255, int(abs(v) * 25))
                    return [intensidad, 100, 255 - intensidad, 160]
                except:
                    return [200, 200, 200, 160] # Gris si hay error

            # 2. Verificamos que las columnas existan
            if 'K' in df_raw.columns and 'Latitud' in df_raw.columns:
                df_mapa = df_raw.copy()
                df_mapa['color'] = df_mapa['K'].apply(color_por_k)
                
                # Tooltip usando 'Tipo_Roca' y 'K'
                t_html = "<b>Roca:</b> {Tipo_Roca}<br><b>K:</b> {K}"

                # 3. Renderizar Mapa
                st.pydeck_chart(pdk.Deck(
                    map_style='mapbox://styles/mapbox/outdoors-v12',
                    initial_view_state=pdk.ViewState(
                        latitude=df_mapa["Latitud"].mean(),
                        longitude=df_mapa["Longitud"].mean(),
                        zoom=12, 
                        pitch=45
                    ),
                    layers=[
                        pdk.Layer(
                            "ScatterplotLayer",
                            df_mapa,
                            get_position=["Longitud", "Latitud"],
                            get_fill_color="color",
                            get_radius=100,
                            pickable=True
                        )
                    ],
                    tooltip={"html": t_html}
                ))

        #****************************************************************************************
        #st.subheader("")
        with st.container(border=True):
            st.subheader("Detalle de Registros")
            st.dataframe(df_raw, use_container_width=True)
            
         #else:
         #       st.error("No se encontró la columna de conductividad (K) en el CSV.")



    
    #****************************************************************************************
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

    #****************************************************************************************
    # --- TAB 4: EXPLICABILIDAD ---
    with tabs[3]:

        st.subheader("Interpretación del Modelo (SHAP)")
        st.info("Esta sección explica qué variables afectan más a la permeabilidad (K).")
        # Aquí llamarías a shap.summary_plot
        # Para Streamlit usa: st.pyplot(plt.gcf()) después de generar el plot de SHAP


else:
    st.warning("👈 Por favor, sube los archivos CSV y TIF en la barra lateral para comenzar.")
    #st.image("https://drive.google.com/uc?export=view&id=1jcdYjLgdkbgomF81QpfbGQYDwePbgIkj", caption="Mi imagen PNG transparente")

