import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

import plotly.express as px
from sklearn.cluster import DBSCAN



def BLOQUE001(): #reescalado
    st.markdown("<h4 style='text-align: center;'>Estandarización de Datos de Pronóstico</h4>", unsafe_allow_html=True)
    st.info("Para que los modelos de Machine Learning predigan correctamente, los datos nuevos de la malla 3D deben ser normalizados usando el mismo escalador exacto del dataset de entrenamiento (K003_scaler_ml).")

    # --- CONTENEDORES DE ENTRADA Y SEGURIDAD ---
    scaler_ml = None
    df_nuevos = None

    col_status1, col_status2 = st.columns(2)

    # 1. VERIFICACIÓN DEL ESCALADOR (K003_scaler_ml)
    with col_status1:
        with st.container(border=True):
            st.markdown("##### 📐 Estado del Escalador")
            if 'K003_scaler_ml' in st.session_state and st.session_state['K003_scaler_ml'] is not None:
                scaler_ml = st.session_state['K003_scaler_ml']
                st.success("✅ Escalador detectado en la memoria activa (K003).")
            else:
                st.warning("⚠️ No se detectó el escalador en memoria.")
                uploaded_scaler = st.file_uploader("Cargar K003_scaler_ml.joblib manualmente:", type=["joblib"], key="upload_scaler_p")
                if uploaded_scaler is not None:
                    scaler_ml = joblib.load(uploaded_scaler)
                    st.success("✅ Escalador cargado desde archivo local.")

    # 2. VERIFICACIÓN DE LOS DATOS NUEVOS (K008_Datos_Nuevos)
    with col_status2:
        with st.container(border=True):
            st.markdown("##### 🧊 Estado de la Malla 3D")
            if 'K008_Datos_Nuevos' in st.session_state and st.session_state['K008_Datos_Nuevos'] is not None:
                df_nuevos = st.session_state['K008_Datos_Nuevos'].copy()
                st.success("✅ Malla 3D detectada en la memoria activa (K008).")
            else:
                st.warning("⚠️ No se detectó la malla 3D en memoria.")
                uploaded_csv = st.file_uploader("Cargar K008_Datos_Nuevos.csv manualmente:", type=["csv"], key="upload_csv_p")
                if uploaded_csv is not None:
                    df_nuevos = pd.read_csv(uploaded_csv)
                    st.success("✅ Datos nuevos cargados desde archivo local.")

    # Frenar la ejecución si falta alguno de los dos pilares
    if scaler_ml is None or df_nuevos is None:
        st.stop()

    # --- 3. PROCESAMIENTO: ESCALADO CON MODELO PREENTRENADO ---
    numerical_cols = ['Profundidad', 'Longitud', 'Latitud', 'Altitud', 'Cota']
    
    st.markdown("---")
    if st.button("⚖️ Aplicar Estandarización a Malla de Pronóstico", use_container_width=True):
        with st.spinner("Transformando variables geométricas al espacio abstracto Z-score..."):
            try:
                # Verificar que las columnas existan en los nuevos datos
                missing_cols = [col for col in numerical_cols if col not in df_nuevos.columns]
                if missing_cols:
                    st.error(f"❌ Error: El dataset ingresado no cuenta con las columnas requeridas: {missing_cols}")
                    st.stop()

                # ATENCIÓN: Usamos .transform() y NO .fit_transform() porque NO queremos calcular una nueva media,
                # sino mapear la nueva geometría bajo las reglas exactas de la data original.
                df_nuevos[numerical_cols] = scaler_ml.transform(df_nuevos[numerical_cols])

                # Guardamos el resultado en el Session State bajo la clave solicitada K009
                st.session_state['K009_reescalado'] = df_nuevos

                st.success("🎉 Datos de pronóstico reescalados con éxito y almacenados en K009_reescalado.")

            except Exception as e:
                st.error(f"❌ Error durante la transformación matemática: {e}")

    # --- 4. INTERFAZ DE USUARIO Y DESCARGAS ---
    if 'K009_reescalado' in st.session_state:
        df_resultado = st.session_state['K009_reescalado']
        
        with st.container(border=True):
            st.markdown("#### 🔍 Inspección de Datos Estandarizados (K009)")
            st.write(f"**Dimensiones del cubo de bloques:** {df_resultado.shape[0]:,} filas × {df_resultado.shape[1]} columnas.")
            
            # Vista previa resumida
            st.dataframe(df_resultado.head(10), use_container_width=True)

            # Generación de la descarga física
            csv_data = df_resultado.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Descargar K009_reescalado.csv",
                data=csv_data,
                file_name="K009_reescalado.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.caption("🚀 **Flujo completado:** Esta matriz normalizada está lista para ser consumida directamente por los modelos entrenados de XGBoost / Random Forest en el siguiente módulo de inferencia litológica.")








def BLOQUE002(): #DBSCAN_Pronostico
    st.markdown("<h4 style='text-align: center;'>Segmentación Geométrica DBSCAN (Datos Nuevos)</h4>", unsafe_allow_html=True)
    st.info("Este módulo ejecuta el agrupamiento por densidad sobre la malla 3D de pronóstico ya normalizada, identificando las estructuras HGS y aislando el ruido geológico (Zonas Complejas) antes de la clasificación litológica final.")

    # --- 1. VERIFICACIÓN Y CARGA DE DATOS (Entrada: K009_reescalado) ---
    df_scaled = None

    if 'K009_reescalado' in st.session_state and st.session_state['K009_reescalado'] is not None:
        df_scaled = st.session_state['K009_reescalado'].copy()
        st.success("✅ Matriz estandarizada detectada en la memoria activa (K009_reescalado).")
    else:
        st.warning("⚠️ No se detectaron los datos reescalados (K009) en la memoria de la sesión.")
        uploaded_csv = st.file_uploader("Cargar K009_reescalado.csv manualmente:", type=["csv"], key="upload_k009_p")
        if uploaded_csv is not None:
            df_scaled = pd.read_csv(uploaded_csv)
            st.success("✅ Matriz cargada con éxito desde el archivo local.")

    # Si no hay datos disponibles, pausamos la ejecución del bloque
    if df_scaled is None:
        st.stop()

    # --- 2. PARAMETRIZACIÓN DEL ALGORITMO (Controles Laterales Dedicados) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Configuración DBSCAN - Datos Nuevos")
    
    # Se usan llaves únicas para no interferir con los controles de la data de entrenamiento original
    val_eps = st.sidebar.slider("Radio de vecindad (eps)", 0.1, 5.0, 0.5, step=0.1, key="dbscan_eps_pronostico")
    val_min_samples = st.sidebar.slider("Muestras mínimas (min_samples)", 1, 20, 5, key="dbscan_min_pronostico")

    # --- 3. PROCESAMIENTO DE CLUSTERING ---
    features_cols = ['Profundidad', 'Longitud', 'Latitud', 'Altitud']

    if all(col in df_scaled.columns for col in features_cols):
        # Extraemos la matriz numérica ya escalada que viene de K009
        features_scaled = df_scaled[features_cols].values

        # Inicialización y ajuste del modelo matemático directo
        dbscan = DBSCAN(eps=val_eps, min_samples=val_min_samples)
        df_scaled['HGS'] = dbscan.fit_predict(features_scaled)

        # Mapeo estándar de anomalías o ruido (-1) a código geológico 999
        df_scaled['HGS'] = df_scaled['HGS'].replace(-1, 999)
        df_scaled['Es_Zona_Compleja'] = (df_scaled['HGS'] == 999).astype(int)

        # --- 4. PANEL DE MÉTRICAS ---
        with st.container(border=True):
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Bloques 3D", f"{len(df_scaled):,}")
            m2.metric("Estructuras HGS Halladas", len(df_scaled[df_scaled['HGS'] != 999]['HGS'].unique()))
            m3.metric("Nodos en Zona Compleja (Ruido)", f"{int(df_scaled['Es_Zona_Compleja'].sum()):,}", delta_color="inverse")

        # --- 5. VISUALIZACIÓN GRÁFICA INTERACTIVA 3D ---
        with st.container(border=True):
            st.markdown("<h5 style='text-align: center;'>Distribución de Estructuras HGS en el Bloque Nuevo</h5>", unsafe_allow_html=True)
            
            df_scaled['Etiqueta'] = df_scaled['HGS'].apply(lambda x: "Zona Compleja" if x == 999 else f"Cluster {x}")

            # Submuestreo estadístico inteligente para asegurar la fluidez de renderizado en el navegador web
            limite_puntos = 25000
            if len(df_scaled) > limite_puntos:
                df_plot = df_scaled.sample(n=limite_puntos, random_state=42)
                st.caption(f"💡 Visualización limitada mediante una muestra aleatoria representativa de {limite_puntos:,} puntos para optimizar la velocidad.")
            else:
                df_plot = df_scaled

            fig = px.scatter_3d(
                df_plot,
                x='Longitud',
                y='Latitud',
                z='Altitud',
                color='Etiqueta',
                symbol='Es_Zona_Compleja',
                opacity=0.6,
                color_discrete_sequence=px.colors.qualitative.Safe,
                labels={'Etiqueta': 'Clasificación Espacial'}
            )

            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=10),
                scene_dragmode='orbit',
                height=600,
                legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.1)
            )

            st.plotly_chart(fig, use_container_width=True)

        # --- 6. PERSISTENCIA DE DATOS Y BOTÓN DE EXPORTACIÓN (Salida: K010_hgs) ---
        # Guardamos la estructura resultante clonada en la memoria global de Streamlit
        st.session_state['K010_hgs'] = df_scaled.copy()

        with st.container(border=True):
            st.markdown("##### 📂 Exportación de Geometría de Subsuelo")
            st.success("✨ Matriz estructurada guardada en el estado del sistema como **K010_hgs**.")
            
            col_h, col_c = st.columns(2)
            
            # Opción 1: Guardar el mapa HTML interactivo generado por Plotly
            html_bytes = fig.to_html().encode('utf-8')
            col_h.download_button(
                label="🌐 Guardar Escena Interactiva HTML",
                data=html_bytes,
                file_name="PROSPECTO_Malla_3D_HGS.html",
                mime="text/html",
                use_container_width=True
            )

            # Opción 2: Descargar el archivo CSV de salida estandarizado K010
            csv_data = df_scaled.to_csv(index=False).encode('utf-8')
            col_c.download_button(
                label="📥 Descargar K010_hgs.csv",
                data=csv_data,
                file_name="K010_hgs.csv",
                mime="text/csv",
                use_container_width=True
            )
            
    else:
        st.error(f"❌ Error Estructural: La matriz cargada desde K009 no cuenta con las columnas requeridas: {features_cols}")






def BLOQUE003(): #Modelamiento litologico
    st.markdown("<h4 style='text-align: center;'>Inferencia Litológica Final con XGBoost</h4>", unsafe_allow_html=True)
    st.info("Este módulo utiliza el clasificador XGBoost entrenado originalmente para predecir de manera tridimensional el Tipo de Roca en cada bloque de la nueva malla mapeada.")

    # --- 1. CONFIGURACIÓN DE ENTRADAS (Carga Dual: Memoria / Archivos Locales) ---
    model_xgb = None
    label_encoder = None
    df_k010 = None

    c1, c2, c3 = st.columns(3)

    # A. Carga del Modelo XGBoost (K005)
    with c1:
        with st.container(border=True):
            st.markdown("##### 🧠 Modelo Clasificador")
            if 'modelo_entrenado_roca' in st.session_state and st.session_state['modelo_entrenado_roca'] is not None:
                model_xgb = st.session_state['modelo_entrenado_roca']
                st.success("✅ XGBoost detectado en memoria.")
            else:
                st.warning("⚠️ Modelo no detectado.")
                uploaded_model = st.file_uploader("Cargar Modelo (.joblib):", type=["joblib"], key="u_model_final")
                if uploaded_model is not None:
                    model_xgb = joblib.load(uploaded_model)
                    st.success("✅ Cargado localmente.")

    # B. Carga del Label Encoder (K005)
    with c2:
        with st.container(border=True):
            st.markdown("##### 🏷️ Decodificador de Clases")
            if 'label_encoder_roca' in st.session_state and st.session_state['label_encoder_roca'] is not None:
                label_encoder = st.session_state['label_encoder_roca']
                st.success("✅ Encoder detectado en memoria.")
            else:
                st.warning("⚠️ Encoder no detectado.")
                uploaded_le = st.file_uploader("Cargar Encoder (.joblib):", type=["joblib"], key="u_le_final")
                if uploaded_le is not None:
                    label_encoder = joblib.load(uploaded_le)
                    st.success("✅ Cargado localmente.")

    # C. Carga de Datos Estructurados de Malla (K010)
    with c3:
        with st.container(border=True):
            st.markdown("##### 🧊 Malla Estructurada")
            if 'K010_hgs' in st.session_state and st.session_state['K010_hgs'] is not None:
                df_k010 = st.session_state['K010_hgs'].copy()
                st.success("✅ Datos K010 detectados en memoria.")
            else:
                st.warning("⚠️ Malla K010 no detectada.")
                uploaded_csv = st.file_uploader("Cargar K010_hgs.csv:", type=["csv"], key="u_csv_final")
                if uploaded_csv is not None:
                    df_k010 = pd.read_csv(uploaded_csv)
                    st.success("✅ Cargado localmente.")

    # Pausar si falta algún prerrequisito
    if model_xgb is None or label_encoder is None or df_k010 is None:
        st.stop()

    # --- 2. PREPARACIÓN DE LAS MATRICES DE INFERENCIA ---
    # Recuperamos exactamente la lista de características numéricas con las que se entrenó el XGBoost
    # Es vital excluir las variables objetivos, auxiliares o cadenas de texto.
    cols_excluir = ['Tipo_Roca', 'log10_K', 'HGS', 'Es_Zona_Compleja', 'Etiqueta']
    features_predict = [col for col in df_k010.columns if col not in cols_excluir]
    
    # Asegurar que solo entren variables numéricas (coordenadas escaladas Z-score)
    X_new = df_k010[features_predict].select_dtypes(include=[np.number])

    # --- 3. EJECUCIÓN DEL PRONÓSTICO ---
    st.markdown("---")
    if st.button("🔮 Ejecutar Pronóstico de Litología 3D", use_container_width=True):
        with st.spinner("Calculando probabilidades multiclase y asignando tipos de roca..."):
            try:
                # Inferencia con XGBoost (devuelve códigos numéricos entrenados)
                y_pred_encoded = model_xgb.predict(X_new)

                # Decodificación inversa para recuperar los nombres reales de las rocas (ej: "Arenisca", "Caliza")
                df_k010['Tipo_Roca_Predicho'] = label_encoder.inverse_transform(y_pred_encoded)

                # Control geológico: Si DBSCAN determinó que el nodo es Ruido Extremo (999), 
                # podemos optar por forzarlo a mantener una etiqueta de incertidumbre o evaluar su predicción.
                # En este pipeline agregamos una etiqueta limpia combinada.
                df_k010['Litologia_Final'] = np.where(df_k010['Es_Zona_Compleja'] == 1, 
                                                      df_k010['Tipo_Roca_Predicho'] + " (Zona Compleja)", 
                                                      df_k010['Tipo_Roca_Predicho'])

                # Guardamos el DataFrame final consolidado en la sesión
                st.session_state['K011_Nuevos_Datos_Pronostico'] = df_k010

                st.success("🎉 ¡Proceso de Inferencia completo! Resultados listos en K011_Nuevos_Datos_Pronostico.")

            except Exception as e:
                st.error(f"❌ Error durante la predicción de XGBoost: {e}")
                st.info("Asegúrate de que las variables escaladas de la malla coincidan con las variables usadas en el entrenamiento.")

    # --- 4. DESPLIEGUE DE RESULTADOS Y VISUALIZACIONES ---
    if 'K011_Nuevos_Datos_Pronostico' in st.session_state:
        df_final = st.session_state['K011_Nuevos_Datos_Pronostico']

        col_g1, col_g2 = st.columns(2)

        # Gráfico 1: Proporción Litológica Predicha
        with col_g1:
            with st.container(border=True):
                st.write("**Resumen de Rocas Predichas en el Volumen 3D**")
                fig_pie = px.pie(df_final, names='Tipo_Roca_Predicho', hole=0.4,
                                 template="plotly_white", color_discrete_sequence=px.colors.qualitative.Safe)
                fig_pie.update_layout(height=350, margin=dict(t=10, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)

        # Gráfico 2: Modelo de bloques predictivo en 3D
        with col_g2:
            with st.container(border=True):
                st.write("**Muestra del Modelo de Bloques Litológico**")
                # Submuestreo para evitar saturar la memoria gráfica del navegador
                df_plot = df_final.sample(n=min(15000, len(df_final)), random_state=42)
                
                fig_scatter = px.scatter_3d(
                    df_plot, x='Longitud', y='Latitud', z='Altitud',
                    color='Tipo_Roca_Predicho', opacity=0.6,
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_scatter.update_layout(height=350, margin=dict(l=0, r=0, b=0, t=10), scene_dragmode='orbit')
                st.plotly_chart(fig_scatter, use_container_width=True)

        # --- 5. EXPORTACIÓN DEL PRODUCTO FINAL ---
        with st.container(border=True):
            st.markdown("##### 📂 Descarga de Resultados Consolidados")
            st.dataframe(df_final.head(10), use_container_width=True)

            # Preparación del CSV K011 solicitado
            csv_final = df_final.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Descargar K011_Nuevos_Datos_Pronostico.csv",
                data=csv_final,
                file_name="K011_Nuevos_Datos_Pronostico.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.caption("🏁 **Pipeline Finalizado con Éxito:** Este archivo contiene las coordenadas geográficas, cotas, profundidades, clasificaciones espaciales por densidad (HGS) y las estimaciones litológicas de inteligencia artificial.")










