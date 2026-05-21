import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def BLOQUE001(df_in_mem): 
    """
    Genera una visualización interactiva en 3D en Streamlit 
    con los valores máximos de log10_K_Predicho.
    
    Parámetros:
    df_in_mem (pd.DataFrame): El DataFrame 'K014_resultado' cargado en memoria.
    """
    st.title("📊 Visualización 3D Interactiva: Valores Máximos")
    st.markdown("Análisis de los valores máximos de `log10_K_Predicho` por punto espacial.")

    # --- CONFIGURACIÓN DE COLUMNAS ---
    st.sidebar.header("Configuración de Coordenadas")
    col_options = df_in_mem.columns.tolist()

    # Selección de ejes para el espacio 3D (se auto-detectan si se llaman X, Y, Z)
    x_axis = st.sidebar.selectbox("Eje X (Longitud / Este)", col_options, index=0 if "X" not in col_options else col_options.index("X"))
    y_axis = st.sidebar.selectbox("Eje Y (Latitud / Norte)", col_options, index=1 if "Y" not in col_options else col_options.index("Y"))
    z_axis = st.sidebar.selectbox("Eje Z (Elevación / Profundidad)", col_options, index=2 if "Z" not in col_options else col_options.index("Z"))

    target_col = "log10_K_Predicho"

    if target_col not in df_in_mem.columns:
        st.error(f"❌ La variable objetivo '{target_col}' no se encuentra en el DataFrame.")
        st.stop()

    # --- PROCESAMIENTO: FILTRAR VALORES MÁXIMOS ---
    # Agrupamos por cada coordenada única (X, Y, Z) y extraemos el valor máximo
    df_max = df_in_mem.groupby([x_axis, y_axis, z_axis])[target_col].max().reset_index()

    # Métricas clave en el panel principal
    col1, col2, col3 = st.columns(3)
    col1.metric("Puntos Únicos Graficados", f"{len(df_max):,}")
    col2.metric("Valor Máximo Absoluto", f"{df_max[target_col].max():.4f}")
    col3.metric("Valor Mínimo Registrado", f"{df_max[target_col].min():.4f}")

    # --- RENDERIZADO DEL GRÁFICO 3D ---
    st.subheader("Espacio Interactivo 3D")

    # Selector de estilo de mapa de color en la barra lateral
    cmap_option = st.sidebar.selectbox(
        "Paleta de Colores",
        ["Viridis", "Plasma", "Inferno", "Magma", "Jet", "Turbo"],
        index=0
    )

    # Creación del objeto gráfico con Plotly
    fig = px.scatter_3d(
        df_max,
        x=x_axis,
        y=y_axis,
        z=z_axis,
        color=target_col,
        color_continuous_scale=cmap_option.lower(),
        labels={target_col: "Máx log10_K"},
        title=f"Dispersión 3D de Máximos para {target_col}",
        opacity=0.85,
        hover_data={x_axis: True, y_axis: True, z_axis: True, target_col: ":.4f"}
    )

    # Ajustes de diseño de la escena interactiva
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            zaxis_title=z_axis,
            aspectmode="data"  # Mantiene la proporción real de las escalas
        ),
        coloraxis_colorbar=dict(title="log10_K")
    )

    # Mostrar el gráfico ocupando todo el ancho de la pantalla
    st.plotly_chart(fig, use_container_width=True)

    # Opcional: Mostrar la tabla de los datos resumidos
    if st.checkbox("Ver tabla de datos resumida"):
        st.dataframe(df_max.sort_values(by=target_col, ascending=False), use_container_width=True)


# --- CÓMO LLAMARLO CORRECTAMENTE EN TU SCRIPT ---
# Asegúrate de descomentar esto abajo para que la función se ejecute pasando la variable en memoria:
# if __name__ == "__main__":
#     BLOQUE001(K014_resultado)

