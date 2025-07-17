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

    df = df[['EDAD', 'GÉNERO', 'CALIFICACIÓN_FINAL', 'AÑO_POSTULA', 'ALCANZO_VACANTE']].dropna()
    df = pd.get_dummies(df, columns=['GÉNERO'], drop_first=False)  # No elimines categorías
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

