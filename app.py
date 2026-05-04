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


#***************************************************************************
logo = Image.open("LOGO_NEREUS.png")
st.set_page_config(page_title="NEREUS App", page_icon=logo)
# 2. Crear tres columnas
# La proporción [1, 1, 1] crea tres espacios iguales. 
# Pondremos el logo en la columna del medio.
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(logo, width=300)
#***************************************************************************



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
st.title("Sistema Inteligente de Gestión de Acuíferos")

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
        # --- 03. COMPARACION LITOLOGICA ---
        st.header("03. COMPARACIÓN LITOLÓGICA: DATOS DE CAMPO VS. REFERENCIA CIENTÍFICA")

        # 1. Biblioteca de Rangos Científicos Típicos (log K en cm/s)
        biblioteca_rangos = {
              'Quartzite': (-11, -8),
              'Granitic Gneiss': (-10, -7),
              'Diorite': (-9, -6),
              'Granodiorite': (-9, -6),
              'Sandstone': (-8, -3),
              'Brecha': (-6, -1),
              'Kyg-Granodiorita': (-9, -6),
              'Monzonite': (-9, -6),
              'Monzodiorite': (-9, -6),
              'Granodiorite-Brecha': (-7, -3),
              'Monzodiorita-Falla': (-5, -1)
       }

       # 2. Calcular mediana para ordenar (Usamos df_raw que es tu variable cargada)
       if 'log10_K' in df_raw.columns and 'Tipo_Roca' in df_raw.columns:
               orden_rocas = df_raw.groupby('Tipo_Roca')['log10_K'].median().sort_values().index

               # 3. Configurar el gráfico con Matplotlib
               fig_comp, ax = plt.subplots(figsize=(12, 8))
               sns.set_theme(style="whitegrid")

               # 4. Crear el Boxplot
               sns.boxplot(
                    data=df_raw, x='log10_K', y='Tipo_Roca', order=orden_rocas,
                    palette="viridis", hue='Tipo_Roca', width=0.6, legend=False, ax=ax
               )

               # 5. Dibujar los rangos científicos (Barras Magenta)
               for i, roca in enumerate(orden_rocas):
                   if roca in biblioteca_rangos:
                       r_min, r_max = biblioteca_rangos[roca]
                       ax.axhspan(i - 0.35, i + 0.35, facecolor='gray', alpha=0.1)
                       ax.hlines(y=i, xmin=r_min, xmax=r_max, color='magenta',
                       linewidth=6, alpha=0.8,
                      label='Rango Científico Típico' if i == 0 else "")

              # 6. Personalización
              ax.set_title('Datos de Campo vs. Referencia Científica', fontsize=16)
              ax.set_xlabel(r'Conductividad Hidráulica $\log_{10}(K)$ [cm/s]', fontsize=13)
              ax.set_ylabel('Tipo de Roca', fontsize=13)

              # Media Proyecto
              mean_val = df_raw['log10_K'].mean()
              ax.axvline(mean_val, color='red', linestyle='--', lw=1.5, label=f'Media Proyecto: {mean_val:.2f}')

              # Leyenda única
              handles, labels = ax.get_legend_handles_labels()
              by_label = dict(zip(labels, handles))
              ax.legend(by_label.values(), by_label.keys(), loc='lower right')

    # 7. Mostrar en Streamlit
    st.pyplot(fig_comp)
    
    # Opcional: Mostrar explicación técnica
    with st.expander("Ver interpretación del gráfico"):
        st.write("""
            * Las **barras magenta** representan los valores reportados en la literatura técnica.
            * Las **cajas (boxplot)** muestran los datos medidos en campo.
            * Si la caja se solapa con la barra magenta, el dato de campo es consistente con la referencia.
        """)
else:
    st.error("Asegúrate de que el CSV tenga las columnas 'log10_K' y 'Tipo_Roca'")


    
 

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
