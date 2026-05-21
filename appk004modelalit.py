import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import plotly.express as px
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold

# --- FUNCIONES CACHEADAS PARA VELOCIDAD ---

@st.cache_data
def filtrar_datos_roca(df):
    conteo_rocas = df['Tipo_Roca'].value_counts()
    # Mantenemos el límite para asegurar suficiencia estadística en los splits
    rocas_suficientes = conteo_rocas[conteo_rocas >= 5].index 
    df_filtrado = df[df['Tipo_Roca'].isin(rocas_suficientes)].copy()
    rocas_eliminadas = list(set(df['Tipo_Roca'].unique()) - set(rocas_suficientes))
    return df_filtrado, rocas_eliminadas

@st.cache_resource
def entrenar_modelo_clasificacion(X_train, y_train, _X_val_cruzada, _y_val_cruzada):
    """
    Entrena el modelo XGBoost y calcula la validación cruzada una sola vez.
    El guion bajo en los argumentos previene que Streamlit intente hashear la matriz completa.
    """
    model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        random_state=42,
        eval_metric='mlogloss'
    )
    
    # Validación Cruzada
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=42)
    cv_scores = cross_val_score(model, _X_val_cruzada, _y_val_cruzada, cv=cv, scoring='accuracy')
    
    # Entrenamiento final
    model.fit(X_train, y_train)
    
    return model, cv_scores

def BLOQUE001():
    st.markdown("<h4 style='text-align: center;'>Modelado de Clasificación: Tipo de Roca</h4>", unsafe_allow_html=True)
    
    # 1. VERIFICAR DATOS DE LA SESIÓN (Entrada corregida: K004_hgs)
    if 'K004_hgs' not in st.session_state or st.session_state['K004_hgs'] is None:
        st.warning("⚠️ No se encontraron los datos procesados por HGS (K004_hgs).")
        st.info("Por favor, completa primero la etapa de **Ingeniería de Rocas por Localización (K004)**.")
        return

    df = st.session_state['K004_hgs'].copy()

    # Filtrado de clases con datos insuficientes (usando caché)
    df_filtrado, rocas_eliminadas = filtrar_datos_roca(df)
    
    if rocas_eliminadas:
        with st.container(border=True): 
            st.warning(f"🔔 Se omitieron rocas con datos insuficientes para splits estratificados: {rocas_eliminadas}")

    # 2. SELECCIÓN DE VARIABLES Y CODIFICACIÓN
    # Excluimos las columnas objetivos o las generadas en procesos paralelos que puedan sesgar al modelo
    cols_objetivo = ['Tipo_Roca', 'log10_K', 'HGS', 'Es_Zona_Compleja', 'Etiqueta']
    X = df_filtrado.drop(columns=[c for c in cols_objetivo if c in df_filtrado.columns])
    X = X.select_dtypes(include=[np.number])
    y = df_filtrado['Tipo_Roca']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 3. DIVISION DE DATOS Y ENTRENAMIENTO
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        with st.status("🧠 Entrenando Clasificador XGBoost...", expanded=True) as status:
            # Llamada al proceso cacheado
            model, cv_scores = entrenar_modelo_clasificacion(X_train, y_train, X, y_encoded)
            status.update(label="✅ Clasificador listo en memoria", state="complete")

            # --- ALMACENAMIENTO EN SESSION STATE PARA PREDICCIONES FUTURAS ---
            st.session_state['modelo_entrenado_roca'] = model
            st.session_state['label_encoder_roca'] = le
            
            status.update(label="✅ Modelo y Encoder vinculados correctamente", state="complete")

        # 4. DESPLIEGUE DE RESULTADOS VISUALES
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.metric("Precisión Media (CV)", f"{cv_scores.mean():.2%}")
            with st.container(border=True):
                st.write("**Importancia de Variables (Feature Importance)**")
                importancias = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
                importancias = importancias.sort_values(by='Importance', ascending=False).head(10)
                fig_imp = px.bar(importancias, x='Importance', y='Feature', orientation='h', color='Importance',
                                 template="plotly_white", color_discrete_sequence=px.colors.qualitative.Safe)
                fig_imp.update_layout(height=300, margin=dict(t=10, b=10))
                st.plotly_chart(fig_imp, use_container_width=True)

        with col2:
            with st.container(border=True):
                st.write("**Distribución de Clases Validadas**")
                fig_pie = px.pie(df_filtrado, names='Tipo_Roca', hole=0.4,
                                 template="plotly_white", color_discrete_sequence=px.colors.qualitative.Safe)
                fig_pie.update_layout(height=350, margin=dict(t=10, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)

        # 5. CONTENEDOR DE EXPORTACIÓN Y COPIAS LOCALES
        with st.container(border=True):  
            st.markdown("##### 📂 Exportación de Artefactos de IA")
            st.success("✨ Los parámetros se han inyectado en la sesión activa para el módulo de Predicción de Litologías.")
            
            c1, c2 = st.columns(2)
            
            # Serialización en memoria del Modelo XGBoost
            model_buf = io.BytesIO()
            joblib.dump(model, model_buf)
            c1.download_button(
                label="💾 Descargar Copia del Modelo (.joblib)", 
                data=model_buf.getvalue(), 
                file_name="K005_modelo_tipo_roca.joblib",
                use_container_width=True
            )

            # Serialización en memoria del LabelEncoder
            le_buf = io.BytesIO()
            joblib.dump(le, le_buf)
            c2.download_button(
                label="🏷️ Descargar Copia del Encoder (.joblib)", 
                data=le_buf.getvalue(), 
                file_name="K005_label_encoder_roca.joblib",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"❌ Ocurrió un error inesperado durante el entrenamiento: {e}")




