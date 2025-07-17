
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(layout="wide")
st.title("🎓 Predicción de Calificación - Admisión UNI")
# Cargar datos
@st.cache_data
def cargar_datos():
    df = pd.read_csv('LIMPIO_ADMISION_UNI_OFICIAL0.csv', encoding='latin1', sep=';')
    
    # Convertir a tipos adecuados

    df['EDAD'] = 2024 - df['AÑO_NACIMIENTO']
    df['CALIFICACIÓN_FINAL'] = pd.to_numeric(df['CALIFICACIÓN_FINAL'], errors='coerce')
    df['AÑO_POSTULA'] = pd.to_numeric(df['AÑO_POSTULA'], errors='coerce')
    df = df[['EDAD', 'GÉNERO', 'CALIFICACIÓN_FINAL', 'AÑO_POSTULA', 'ALCANZO_VACANTE', 'MODALIDAD']].dropna()
    df = pd.get_dummies(df, columns=['GÉNERO'], drop_first=False)
    return df

df = cargar_datos()

# ----------- SECCIÓN 1: Predicción individual -----------

st.header("🔮 Predicción de Calificación por Edad y Género")
X = df[['EDAD', 'GÉNERO_MASCULINO']] if 'GÉNERO_MASCULINO' in df.columns else df[['EDAD']]
y = df['CALIFICACIÓN_FINAL']
modelo = LinearRegression()
modelo.fit(X, y)

edad = st.slider("📅 Edad del postulante", 15, 40, 18)
genero = st.selectbox("⚧️ Género", ["MASCULINO", "FEMENINO"])
genero_val = 1 if genero == "MASCULINO" else 0

if 'GÉNERO_MASCULINO' in df.columns:
    entrada = pd.DataFrame([[edad, genero_val]], columns=['EDAD', 'GÉNERO_MASCULINO'])
else:
    entrada = pd.DataFrame([[edad]], columns=['EDAD'])

pred = modelo.predict(entrada)[0]
st.metric("🎯 Calificación estimada", f"{pred:.2f}")

# ----------- SECCIÓN 2: Distribución de Calificaciones -----------

st.subheader("📊 Distribución de Calificaciones")
fig1, ax1 = plt.subplots()
df['CALIFICACIÓN_FINAL'].hist(ax=ax1, bins=20)
st.pyplot(fig1)

# ----------- SECCIÓN 3: Evolución por Año y Género -----------

if 'AÑO_POSTULA' in df.columns:
    st.subheader("📈 Evolución de Postulantes por Año y Género")
    cols = [col for col in df.columns if 'GÉNERO_' in col]
    conteo = df.groupby(['AÑO_POSTULA'])[cols].sum()
    conteo.columns = [c.replace('GÉNERO_', '') for c in conteo.columns]
    st.line_chart(conteo)

    # ----------- SECCIÓN 4: Proyección futura total -----------
    
    st.subheader("📅 Proyección Futura de Postulantes")
    total_por_año = df.groupby('AÑO_POSTULA').size()
    años = total_por_año.index.values.reshape(-1, 1)
    postulantes = total_por_año.values.reshape(-1, 1)

    modelo_futuro = LinearRegression()
    modelo_futuro.fit(años, postulantes)

    futuros = np.arange(años.max()+1, años.max()+4).reshape(-1, 1)
    pred_futuros = modelo_futuro.predict(futuros)

    df_pred = pd.DataFrame(pred_futuros, index=futuros.flatten(), columns=["Proyectado"])
    st.line_chart(pd.concat([total_por_año, df_pred.squeeze()], axis=0))

# ----------- SECCIÓN 5: Comparación de Calificación por Género -----------

st.subheader("⚖️ Comparación de Calificaciones por Género")

# Asegurar que GÉNERO esté presente para gráfico
if 'GÉNERO_MASCULINO' in df.columns:
    df['GÉNERO_LABEL'] = df['GÉNERO_MASCULINO'].apply(lambda x: "MASCULINO" if x == 1 else "FEMENINO")
else:
    df['GÉNERO_LABEL'] = df['GÉNERO_FEMENINO'].apply(lambda x: "FEMENINO" if x == 1 else "MASCULINO")

fig2, ax2 = plt.subplots()
sns.boxplot(x='GÉNERO_LABEL', y='CALIFICACIÓN_FINAL', data=df, ax=ax2)
st.pyplot(fig2)

# ----------- SECCIÓN 6: Análisis por MODALIDAD -----------

if 'MODALIDAD' in df.columns:
    st.subheader("🎓 Análisis por Modalidad de Postulación")

# Distribución de calificación por modalidad
    
    st.markdown("**Comparación de calificaciones por modalidad**")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sns.boxplot(x='MODALIDAD', y='CALIFICACIÓN_FINAL', data=df, ax=ax3)
    ax3.tick_params(axis='x', rotation=45)
    st.pyplot(fig3)

        # Conteo de postulantes por modalidad y año

    if 'AÑO_POSTULA' in df.columns:
        st.markdown("**Evolución de postulantes por modalidad y año**")
        modalidad_anio = df.groupby(['AÑO_POSTULA', 'MODALIDAD']).size().unstack(fill_value=0)
        st.line_chart(modalidad_anio)

        # Proyección futura para cada modalidad (opcional básico)
        
        st.markdown("**Selecciona una modalidad para proyectar su crecimiento:**")
        opcion_modalidad = st.selectbox("Modalidades disponibles:", modalidad_anio.columns.tolist())
        
        # Datos de la modalidad seleccionada
        serie = modalidad_anio[opcion_modalidad].reset_index()
        X_mod = serie[['AÑO_POSTULA']]
        y_mod = serie[opcion_modalidad]
        
        if len(X_mod) >= 2:
            modelo_mod = LinearRegression()
            modelo_mod.fit(X_mod, y_mod)
        
            años_futuros = np.arange(X_mod['AÑO_POSTULA'].max() + 1, X_mod['AÑO_POSTULA'].max() + 4).reshape(-1, 1)
            pred = modelo_mod.predict(años_futuros)
        
            df_pred = pd.Series(pred, index=años_futuros.flatten(), name='Proyección')
            grafico = pd.concat([serie.set_index('AÑO_POSTULA')[opcion_modalidad], df_pred])
        
            st.line_chart(grafico)
        else:
            st.warning("No hay suficientes datos para proyectar esta modalidad.")

        # ----------- SECCIÓN 7: Análisis por Departamento y Distrito del Colegio -----------

        if 'COLEGIO_DEPARTAMENTO' in df.columns:
            st.subheader("🗺️ Análisis por Departamento del Colegio")
        
            # Cantidad de postulantes por departamento
            dept_counts = df['COLEGIO_DEPARTAMENTO'].value_counts().sort_values(ascending=False)
            st.bar_chart(dept_counts)
        
            # Promedio de calificación por departamento
            avg_by_dept = df.groupby('COLEGIO_DEPARTAMENTO')['CALIFICACIÓN_FINAL'].mean().sort_values(ascending=False)
            st.subheader("🎓 Promedio de Calificaciones por Departamento")
            st.bar_chart(avg_by_dept)
        
        if 'COLEGIO_DISTRITO' in df.columns:
            st.subheader("🏘️ Análisis por Distrito del Colegio")
        
            top_distritos = df['COLEGIO_DISTRITO'].value_counts().head(10).index.tolist()
            df_top = df[df['COLEGIO_DISTRITO'].isin(top_distritos)]
        
            # Calificaciones por distrito (boxplot solo para los 10 distritos con más postulantes)
            fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
            sns.boxplot(x='COLEGIO_DISTRITO', y='CALIFICACIÓN_FINAL', data=df_top, ax=ax_dist)
            ax_dist.tick_params(axis='x', rotation=45)
            st.pyplot(fig_dist)


