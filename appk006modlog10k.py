import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold

# --- FUNCIONES CACHEADAS PARA VELOCIDAD ---

@st.cache_data
def transformar_y_dummies(df):
    """
    Separa las características (X) y el objetivo (y), convirtiendo 
    la columna 'Tipo_Roca' en variables dummies numéricas.
    """
    # 1. Variable objetivo para Regresión (Sin escalar)
    y = df['log10_K'].copy()
    
    # 2. Definir columnas a eliminar para construir las variables de entrada (X)
    cols_eliminar = ['log10_K', 'Etiqueta']
    X_base = df.drop(columns=[c for c in cols_eliminar if c in df.columns])
    
    # 3. Aplicar One-Hot Encoding (Dummies) a la columna Tipo_Roca si existe
    if 'Tipo_Roca' in X_base.columns:
        X_encoded = pd.get_dummies(X_base, columns=['Tipo_Roca'], drop_first=False, dtype=int)
    else:
        X_encoded = X_base.select_dtypes(include=[np.number])
        
    return X_encoded, y

@st.cache_resource
def entrenar_modelo_regresion(X_train, y_train, _X_val_cruzada, _y_val_cruzada):
    """
    Entrena el modelo XGBRegressor y calcula la validación cruzada (R²) una sola vez.
    """
    model = XGBRegressor(
        n_estimators=100, 
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )
    
    # Validación Cruzada usando R² para regresión
    cv = RepeatedKFold(n_splits=3, n_repeats=1, random_state=42)
    cv_scores = cross_val_score(model, _X_val_cruzada, _y_val_cruzada, cv=cv, scoring='r2')
    
    # Entrenamiento final
    model.fit(X_train, y_train)
    
    return model, cv_scores

def BLOQUE001(): #Modelado_Permeabilidad
    st.markdown("<h4 style='text-align: center;'>Modelado de Regresión: Permeabilidad (log10_K)</h4>", unsafe_allow_html=True)
    st.info("Se entrena un algoritmo XGBRegressor para estimar la permeabilidad. La litología se transforma automáticamente en variables dummies (0 o 1) para actuar como una propiedad física de entrada.")

    # 1. VERIFICAR DATOS DE LA SESIÓN (Entrada: K004_hgs)
    if 'K004_hgs' not in st.session_state or st.session_state['K004_hgs'] is None:
        st.warning("⚠️ No se encontraron los datos estructurados por HGS (K004_hgs).")
        st.info("Por favor, completa primero el procesamiento en la pestaña de **Ingeniería de Rocas (K004)**.")
        return

    df = st.session_state['K004_hgs'].copy()

    # Verificar que la variable objetivo exista en el dataset
    if 'log10_K' not in df.columns:
        st.error("❌ Error crítico: La columna objetivo 'log10_K' no se encuentra en el archivo K004_hgs.")
        return

    # 2. PROCESAMIENTO DE DUMMIES Y SEPARACIÓN MATRICIAL
    X, y = transformar_y_dummies(df)

    # 3. DIVISIÓN DE DATOS Y ENTRENAMIENTO
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        with st.status("🧠 Entrenando Regresor XGBoost para Permeabilidad...", expanded=True) as status:
            model, cv_scores = entrenar_modelo_regresion(X_train, y_train, X, y)
            status.update(label="✅ Regresor listo en memoria", state="complete")

            # --- ALMACENAMIENTO EN SESSION STATE PARA PREDICCIONES FUTURAS ---
            st.session_state['modelo_entrenado_k'] = model
            st.session_state['columnas_dummies_k'] = X.columns.tolist()
            
            status.update(label="✅ Modelo y Esquema de Dummies vinculados correctamente", state="complete")

        # 4. DESPLIEGUE DE RESULTADOS VISUALES
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                st.metric("Bondad de Ajuste Media ($R^2$ CV)", f"{cv_scores.mean():.2f}")
                st.caption("Un valor cercano a 1.0 indica un ajuste predictivo óptimo de la permeabilidad.")
                
            with st.container(border=True):
                st.write("**Importancia de Variables (Features de Entrada + Dummies)**")
                importancias = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
                importancias = importancias.sort_values(by='Importance', ascending=False).head(10)
                
                fig_imp = px.bar(importancias, x='Importance', y='Feature', orientation='h', color='Importance',
                                 template="plotly_white", color_discrete_sequence=px.colors.qualitative.Bold)
                fig_imp.update_layout(height=300, margin=dict(t=10, b=10))
                st.plotly_chart(fig_imp, use_container_width=True)

        with col2:
            with st.container(border=True):
                st.write("**Análisis de Residuos (Predicho vs Real)**")
                y_pred_test = model.predict(X_test)
                df_residuos = pd.DataFrame({'Real': y_test, 'Predicho': y_pred_test})
                
                fig_scatter = px.scatter(df_residuos, x='Real', y='Predicho', trendline="ols",
                                         labels={'Real': 'log10_K Real', 'Predicho': 'log10_K Predicho'},
                                         template="plotly_white", color_discrete_sequence=['#1f77b4'])
                fig_scatter.update_layout(height=380, margin=dict(t=10, b=10))
                st.plotly_chart(fig_scatter, use_container_width=True)

        # 5. CONTENEDOR DE EXPORTACIÓN (Nombres de archivos corregidos)
        with st.container(border=True):  
            st.markdown("##### 📂 Exportación de Artefactos de Regresión")
            st.success("✨ Los pesos matemáticos del regresor se han inyectado en la sesión de Streamlit.")
            
            c1, c2 = st.columns(2)
            
            # Serialización en memoria del Modelo XGBRegressor (K013_modelo_xgboost_logK.joblib)
            model_buf = io.BytesIO()
            joblib.dump(model, model_buf)
            c1.download_button(
                label="💾 Descargar Regresor K013 (.joblib)", 
                data=model_buf.getvalue(), 
                file_name="K013_modelo_xgboost_logK.joblib",
                use_container_width=True,
                help="Descarga el archivo del modelo entrenado para la predicción del log10_K."
            )

            # Guardamos el esquema estructurado de las columnas dummies (K013_columnas_modelo.joblib)
            cols_buf = io.BytesIO()
            joblib.dump(X.columns.tolist(), cols_buf)
            c2.download_button(
                label="📋 Descargar Esquema de Columnas K013 (.joblib)", 
                data=cols_buf.getvalue(), 
                file_name="K013_columnas_modelo.joblib",
                use_container_width=True,
                help="Guarda el orden y nombres de las columnas para mapear idénticamente los datos de pronóstico."
            )

    except Exception as e:
        st.error(f"❌ Ocurrió un error inesperado durante el entrenamiento del regresor: {e}")







