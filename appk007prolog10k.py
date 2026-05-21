import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

import plotly.express as px
from sklearn.cluster import DBSCAN


def BLOQUE001(): #_Reescalado_Final_Pronostico
    st.markdown("<h4 style='text-align: center;'>Estandarización Final para Modelos de Regresión</h4>", unsafe_allow_html=True)
    st.info("Este módulo aplica el escalador original (K003) sobre las variables geométricas de la nueva malla mapeada que ya incluye las predicciones litológicas (K011), asegurando la compatibilidad matemática con el regresor de permeabilidad.")

    # --- 1. CONFIGURACIÓN DE ENTRADAS (Carga Dual: Memoria / Archivos Locales) ---
    scaler_ml = None
    df_k011 = None

    col_status1, col_status2 = st.columns(2)

    # A. Verificación del Escalador Matemático (K003_scaler_ml)
    with col_status1:
        with st.container(border=True):
            st.markdown("##### 📐 Escalador Original")
            if 'K003_scaler_ml' in st.session_state and st.session_state['K003_scaler_ml'] is not None:
                scaler_ml = st.session_state['K003_scaler_ml']
                st.success("✅ Escalador detectado en la memoria activa (K003).")
            else:
                st.warning("⚠️ Escalador no detectado en memoria.")
                uploaded_scaler = st.file_uploader("Cargar K003_scaler_ml.joblib:", type=["joblib"], key="u_scaler_f")
                if uploaded_scaler is not None:
                    scaler_ml = joblib.load(uploaded_scaler)
                    st.success("✅ Escalador cargado localmente.")

    # B. Verificación de Datos de Pronóstico con Litología (K011)
    with col_status2:
        with st.container(border=True):
            st.markdown("##### 🧊 Malla con Litología")
            if 'K011_Nuevos_Datos_Pronostico' in st.session_state and st.session_state['K011_Nuevos_Datos_Pronostico'] is not None:
                df_k011 = st.session_state['K011_Nuevos_Datos_Pronostico'].copy()
                st.success("✅ Datos K011 detectados en la memoria activa.")
            else:
                st.warning("⚠️ Datos K011 no detectados en memoria.")
                uploaded_csv = st.file_uploader("Cargar K011_Nuevos_Datos_Pronostico.csv:", type=["csv"], key="u_csv_f")
                if uploaded_csv is not None:
                    df_k011 = pd.read_csv(uploaded_csv)
                    st.success("✅ Datos cargados localmente.")

    # Pausar la ejecución si falta algún componente crítico
    if scaler_ml is None or df_k011 is None:
        st.stop()

    # --- 2. PROCESAMIENTO: ESCALADO GEOMÉTRICO ---
    numerical_cols = ['Profundidad', 'Longitud', 'Latitud', 'Altitud', 'Cota']
    
    st.markdown("---")
    if st.button("⚖️ Ejecutar Reescalado de Variables Geométricas", use_container_width=True):
        with st.spinner("Normalizando coordenadas en base al histórico de entrenamiento..."):
            try:
                # Validar la existencia de las columnas requeridas
                missing_cols = [col for col in numerical_cols if col not in df_k011.columns]
                if missing_cols:
                    st.error(f"❌ Error: El dataset ingresado no cuenta con las variables necesarias: {missing_cols}")
                    st.stop()

                # IMPORTANTE: Se usa .transform() para mantener los parámetros de media y desviación estándar originales
                df_k011[numerical_cols] = scaler_ml.transform(df_k011[numerical_cols])

                # Guardar el resultado en la sesión global bajo la clave K012
                st.session_state['K012_Escaleado'] = df_k011

                st.success("🎉 Datos de la malla transformados con éxito y guardados en K012_Escaleado.")

            except Exception as e:
                st.error(f"❌ Error durante la normalización matemática: {e}")

    # --- 3. DESPLIEGUE DE RESULTADOS Y EXPORTACIÓN ---
    if 'K012_Escaleado' in st.session_state:
        df_resultado = st.session_state['K012_Escaleado']
        
        with st.container(border=True):
            st.markdown("#### 🔍 Inspección de Datos Estandarizados (K012)")
            st.write(f"**Registros procesados:** {df_resultado.shape[0]:,} filas × {df_resultado.shape[1]} columnas.")
            
            # Vista previa de la tabla resultante
            st.dataframe(df_resultado.head(10), use_container_width=True)

            # Preparación del flujo de bytes para la descarga del CSV K012 solicitado
            csv_data = df_resultado.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Descargar K012_Escaleado.csv",
                data=csv_data,
                file_name="K012_Escaleado.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.caption("🚀 **Flujo completado:** Este archivo mantiene la integridad de los clusters y las predicciones litológicas, pero con sus coordenadas normalizadas. Está listo para ingresar al regresor XGBoost final (K013) y estimar los valores de permeabilidad.")





def BLOQUE002(): #_DBSCAN_Final_Regresion
    st.markdown("<h4 style='text-align: center;'>Segmentación Geométrica DBSCAN Final</h4>", unsafe_allow_html=True)
    st.info("Este módulo ejecuta un análisis de densidad DBSCAN sobre las coordenadas normalizadas de la malla de predicción (K012). Esto inyecta las variables de entorno 'HGS' y 'Es_Zona_Compleja' requeridas por el regresor XGBoost (K013) para estimar la permeabilidad.")

    # --- 1. VERIFICACIÓN Y CARGA DE DATOS (Entrada: K012_Escaleado) ---
    df_scaled = None

    if 'K012_Escaleado' in st.session_state and st.session_state['K012_Escaleado'] is not None:
        df_scaled = st.session_state['K012_Escaleado'].copy()
        st.success("✅ Matriz estandarizada y con litología detectada en memoria (K012_Escaleado).")
    else:
        st.warning("⚠️ No se encontraron los datos reescalados (K012) en la memoria de la sesión.")
        uploaded_csv = st.file_uploader("Cargar K012_Escaleado.csv manualmente:", type=["csv"], key="u_csv_k012_dbscan")
        if uploaded_csv is not None:
            df_scaled = pd.read_csv(uploaded_csv)
            st.success("✅ Matriz cargada con éxito desde el archivo local.")

    # Si no hay datos disponibles, detenemos la ejecución de este bloque
    if df_scaled is None:
        st.stop()

    # --- 2. PARAMETRIZACIÓN DEL ALGORITMO (Controles Laterales Dedicados) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("DBSCAN - Preparación de Regresión")
    
    # Llaves únicas para evitar conflictos con otros bloques DBSCAN del sistema
    val_eps = st.sidebar.slider("Radio de vecindad (eps)", 0.1, 5.0, 0.5, step=0.1, key="eps_dbscan_regresion")
    val_min_samples = st.sidebar.slider("Muestras mínimas (min_samples)", 1, 20, 5, key="min_dbscan_regresion")

    # --- 3. PROCESAMIENTO DE CLUSTERING ---
    features_cols = ['Profundidad', 'Longitud', 'Latitud', 'Altitud']

    if all(col in df_scaled.columns for col in features_cols):
        # Extraemos la matriz de variables geométricas ya normalizadas en K012
        features_scaled = df_scaled[features_cols].values

        # Ajuste directo del algoritmo sobre el nuevo espacio vectorial
        dbscan = DBSCAN(eps=val_eps, min_samples=val_min_samples)
        df_scaled['HGS'] = dbscan.fit_predict(features_scaled)

        # Mapeo estándar de anomalías de densidad (-1) al código de Zona Compleja (999)
        df_scaled['HGS'] = df_scaled['HGS'].replace(-1, 999)
        df_scaled['Es_Zona_Compleja'] = (df_scaled['HGS'] == 999).astype(int)

        # --- 4. PANEL DE MÉTRICAS EN INTERFAZ ---
        with st.container(border=True):
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Nodos Mapeados", f"{len(df_scaled):,}")
            m2.metric("Estructuras HGS Halladas", len(df_scaled[df_scaled['HGS'] != 999]['HGS'].unique()))
            m3.metric("Puntos en Zona Compleja", f"{int(df_scaled['Es_Zona_Compleja'].sum()):,}", delta_color="inverse")

        # --- 5. VISUALIZACIÓN GRÁFICA INTERACTIVA 3D ---
        with st.container(border=True):
            st.markdown("<h5 style='text-align: center;'>Distribución del Entorno HGS antes de Regresión</h5>", unsafe_allow_html=True)
            
            df_scaled['Etiqueta_HGS'] = df_scaled['HGS'].apply(lambda x: "Zona Compleja" if x == 999 else f"Cluster {x}")

            # Submuestreo inteligente para no congelar la GPU del navegador del usuario
            limite_puntos = 20000
            if len(df_scaled) > limite_puntos:
                df_plot = df_scaled.sample(n=limite_puntos, random_state=42)
                st.caption(f"💡 Renderizado optimizado: mostrando una muestra aleatoria de {limite_puntos:,} puntos.")
            else:
                df_plot = df_scaled

            fig = px.scatter_3d(
                df_plot,
                x='Longitud',
                y='Latitud',
                z='Altitud',
                color='Etiqueta_HGS',
                symbol='Es_Zona_Compleja',
                opacity=0.6,
                color_discrete_sequence=px.colors.qualitative.Bold,
                labels={'Etiqueta_HGS': 'Clasificación HGS'}
            )

            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=10),
                scene_dragmode='orbit',
                height=550
            )

            st.plotly_chart(fig, use_container_width=True)

        # --- 6. PERSISTENCIA DE DATOS Y EXPORTACIÓN (Salida: K013_hgs.csv) ---
        # Guardamos en memoria interna el DataFrame resultante
        st.session_state['K013_hgs'] = df_scaled.copy()

        with st.container(border=True):
            st.markdown("##### 📂 Exportación de Matriz Estructurada (K013)")
            st.success("✨ Dataset guardado de forma interna en `st.session_state['K013_hgs']`.")
            
            col_h, col_c = st.columns(2)
            
            # Opción 1: Exportar el mapa HTML interactivo de Plotly
            html_bytes = fig.to_html().encode('utf-8')
            col_h.download_button(
                label="🌐 Guardar Escena Geométrica HTML",
                data=html_bytes,
                file_name="PROSPECTO_Malla_Regresion_HGS.html",
                mime="text/html",
                use_container_width=True
            )

            # Opción 2: Descargar el archivo CSV de salida K013 solicitado
            # Eliminamos la columna auxiliar de etiquetas para dejar el archivo completamente limpio
            df_descarga = df_scaled.drop(columns=['Etiqueta_HGS'], errors='ignore')
            csv_data = df_descarga.to_csv(index=False).encode('utf-8')
            
            col_c.download_button(
                label="📥 Descargar K013_hgs.csv",
                data=csv_data,
                file_name="K013_hgs.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.caption("🏁 **Paso Completado:** Este dataset contiene ahora las variables geométricas Z-score, la litología predicha por el clasificador y la segmentación espacial por densidad. Está listo para el módulo final de regresión de permeabilidad.")
            
    else:
        st.error(f"❌ Error Estructural: La matriz cargada desde K012 no cuenta con las columnas requeridas: {features_cols}")









def BLOQUE003(): #_Pronostico_Permeabilidad_Final
    st.markdown("<h4 style='text-align: center;'>Inferencia Final: Predicción de Permeabilidad (log10_K)</h4>", unsafe_allow_html=True)
    st.info("Este módulo procesa las variables dummies de la litología predicha bajo el esquema estricto de entrenamiento y ejecuta el regresor XGBoost para estimar la permeabilidad en la nueva malla.")

    # --- 1. CONFIGURACIÓN DE ENTRADAS (Carga Dual: Memoria / Archivos Locales) ---
    df_k013 = None
    columnas_esquema = None
    model_regresor = None

    c1, c2, c3 = st.columns(3)

    # A. Entrada de la Malla Estructurada (K013)
    with c1:
        with st.container(border=True):
            st.markdown("##### 🧊 Malla HGS (K013)")
            if 'K013_hgs' in st.session_state and st.session_state['K013_hgs'] is not None:
                df_k013 = st.session_state['K013_hgs'].copy()
                st.success("✅ Datos K013 en memoria.")
            else:
                st.warning("⚠️ K013 no detectado.")
                uploaded_csv = st.file_uploader("Cargar K013_hgs.csv:", type=["csv"], key="u_k013_reg")
                if uploaded_csv is not None:
                    df_k013 = pd.read_csv(uploaded_csv)
                    st.success("✅ CSV Cargado.")

    # B. Entrada del Esquema de Columnas (K013)
    with c2:
        with st.container(border=True):
            st.markdown("##### 📋 Esquema de Dummies")
            if 'columnas_dummies_k' in st.session_state and st.session_state['columnas_dummies_k'] is not None:
                columnas_esquema = st.session_state['columnas_dummies_k']
                st.success("✅ Esquema en memoria.")
            else:
                st.warning("⚠️ Esquema no detectado.")
                uploaded_cols = st.file_uploader("Cargar K013_columnas_modelo.joblib:", type=["joblib"], key="u_cols_reg")
                if uploaded_cols is not None:
                    columnas_esquema = joblib.load(uploaded_cols)
                    st.success("✅ Esquema Cargado.")

    # C. Entrada del Modelo Regresor XGBoost (K013)
    with c3:
        with st.container(border=True):
            st.markdown("##### 🧠 Regresor LogK")
            if 'modelo_entrenado_k' in st.session_state and st.session_state['modelo_entrenado_k'] is not None:
                model_regresor = st.session_state['modelo_entrenado_k']
                st.success("✅ Regresor en memoria.")
            else:
                st.warning("⚠️ Regresor no detectado.")
                uploaded_model = st.file_uploader("Cargar K013_modelo_xgboost_logK.joblib:", type=["joblib"], key="u_model_reg")
                if uploaded_model is not None:
                    model_regresor = joblib.load(uploaded_model)
                    st.success("✅ Regresor Cargado.")

    # Frenar la ejecución si falta algún componente crítico
    if df_k013 is None or columnas_esquema is None or model_regresor is None:
        st.stop()

    # --- 2. PROCESAMIENTO MIGRATORIO A DUMMIES (Generación de K014_Dummies) ---
    st.markdown("---")
    if st.button("🚀 Procesar Dummies y Pronosticar Permeabilidad", use_container_width=True):
        with st.spinner("Alineando variables y ejecutando algoritmo de regresión..."):
            try:
                # Usamos la columna base 'Tipo_Roca_Predicho' para generar las dummies correspondientes al entrenamiento
                # (Nota: 'Litologia_Final' contiene el texto modificado con "(Zona Compleja)", por lo que 'Tipo_Roca_Predicho' es más limpia)
                col_target_dummy = 'Tipo_Roca_Predicho'
                
                if col_target_dummy not in df_k013.columns:
                    st.error(f"❌ Error Estructural: Falta la columna '{col_target_dummy}' en los datos K013.")
                    st.stop()

                # Generar dummies de la data de entrada actual usando un prefijo estándar si es necesario, 
                # o mapeando directamente según cómo se hayan guardado en K013_columnas_modelo.joblib
                # Asumimos que pd.get_dummies(..., columns=['Tipo_Roca_Predicho']) generará columnas tipo 'Tipo_Roca_Predicho_Arenisca' o similar.
                # Si en tu entrenamiento usaste directamente la columna original sin prefijo, puedes ajustar el comportamiento aquí.
                df_dummies_raw = pd.get_dummies(df_k013, columns=[col_target_dummy], drop_first=False, dtype=int)
                
                # REGLA DE ORO DE INFERENCIA: Reconstruir la matriz exacta del esquema original
                X_final = pd.DataFrame(0, index=df_k013.index, columns=columnas_esquema)
                
                for col in columnas_esquema:
                    if col in df_dummies_raw.columns:
                        X_final[col] = df_dummies_raw[col]
                    elif col in df_k013.columns:
                        # Para variables numéricas directas como Longitud, Latitud, Altitud, HGS, Es_Zona_Compleja
                        X_final[col] = df_k013[col]

                # --- EXPORTACIÓN INTERMEDIA: K014_Dummies.csv ---
                st.session_state['K014_Dummies'] = X_final.copy()

                # --- 3. INFERENCIA CON EL REGRESOR XGBOOST ---
                predicciones_log10_K = model_regresor.predict(X_final)

                # --- COMPILACIÓN DEL PRODUCTO FINAL: K014_resultado.csv ---
                df_k014_resultado = df_k013.copy()
                df_k014_resultado['log10_K_Predicho'] = predicciones_log10_K

                # Guardamos los resultados finales en la sesión activa
                st.session_state['K014_resultado'] = df_k014_resultado

                st.success("🎉 ¡Proceso completado exitosamente! Los archivos K014 están listos para descarga.")

            except Exception as e:
                st.error(f"❌ Error crítico en el alineamiento matricial o inferencia: {e}")

    # --- 4. SECCIÓN DE DESCARGAS DE PRODUCTOS (K014) ---
    if 'K014_Dummies' in st.session_state and 'K014_resultado' in st.session_state:
        df_d = st.session_state['K014_Dummies']
        df_r = st.session_state['K014_resultado']

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            with st.container(border=True):
                st.markdown("##### 🔍 Vista Previa: Matriz Dummies (K014_Dummies)")
                st.dataframe(df_d.head(5), use_container_width=True)
                
                csv_d = df_d.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar K014_Dummies.csv",
                    data=csv_d,
                    file_name="K014_Dummies.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with col_v2:
            with st.container(border=True):
                st.markdown("##### 🔮 Vista Previa: Resultados de Permeabilidad")
                
                # Mapeo explícito y seguro para la visualización del DataFrame final sin KeyErrors
                columnas_muestreo = ['Longitud', 'Latitud', 'Altitud', 'Tipo_Roca_Predicho', 'Litologia_Final', 'log10_K_Predicho']
                columnas_validas = [c for c in columnas_muestreo if c in df_r.columns]
                
                st.dataframe(df_r[columnas_validas].head(5), use_container_width=True)
                
                csv_r = df_r.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar K014_resultado.csv",
                    data=csv_r,
                    file_name="K014_resultado.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        # --- 5. VISUALIZACIÓN EN 3D DE LA PERMEABILIDAD ESTIMADA ---
        with st.container(border=True):
            st.markdown("<h5 style='text-align: center;'>Distribución Tridimensional de Permeabilidad Estimada (log10_K)</h5>", unsafe_allow_html=True)
            
            # Filtro para optimizar el rendimiento del gráfico
            df_plot = df_r.sample(n=min(15000, len(df_r)), random_state=42)
            
            fig = px.scatter_3d(
                df_plot, x='Longitud', y='Latitud', z='Altitud',
                color='log10_K_Predicho',
                color_continuous_scale=px.colors.sequential.Jet,
                opacity=0.7,
                labels={'log10_K_Predicho': 'log10_K Est.'}
            )
            fig.update_layout(height=500, margin=dict(l=0, r=0, b=0, t=10), scene_dragmode='orbit')
            st.plotly_chart(fig, use_container_width=True)




