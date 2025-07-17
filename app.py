import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("🎓 Predicción de Calificación - Admisión UNHEVAL")

# Cargar datos
@st.cache_data
def cargar_datos():
    df = pd.read_csv('LIMPIO_ADMISION_UNI_OFICIAL0.csv', encoding='latin1', sep=';')
    df['EDAD'] = 2024 - df['AÑO_NACIMIENTO']
    df = df[['EDAD', 'GÉNERO', 'CALIFICACIÓN_FINAL']].dropna()
    df = pd.get_dummies(df, columns=['GÉNERO'], drop_first=True)
    return df

df = cargar_datos()

# Modelo
X = df[['EDAD', 'GÉNERO_MASCULINO']]  
y = df['CALIFICACIÓN_FINAL']
modelo = LinearRegression()
modelo.fit(X, y)


# Inputs del usuario
edad = st.slider("📅 Edad del postulante", min_value=15, max_value=40, value=18)
genero = st.selectbox("⚧️ Género", ["MASCULINO", "FEMENINO"])
genero_masculino = 1 if genero == "MASCULINO" else 0  #

entrada = pd.DataFrame([[edad, genero_masculino]], columns=['EDAD', 'GÉNERO_MASCULINO']) 
pred = modelo.predict(entrada)[0]

st.metric("🎯 Calificación estimada", f"{pred:.2f}")

# Visualización
st.subheader("📊 Distribución de Calificaciones")
fig, ax = plt.subplots()
df['CALIFICACIÓN_FINAL'].hist(ax=ax, bins=20)
st.pyplot(fig)
