import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.spatial import distance
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="Proyecto Final - Centro Inteligencia", layout="wide")

st.title("Consola de Inteligencia de la Alianza")

# --- CARGAR LOS DATOS ---
# Uso cache para que no tarde tanto al mover los filtros
@st.cache_data
def cargar_csv():
    datos = pd.read_csv('players_data.csv')
    return datos

df = cargar_csv()

# --- PREPARAR LOS DATOS (NORMALIZACIÓN) ---
# Selecciono las columnas de rendimiento para el algoritmo de similitud
cols_rendimiento = ['Goles', 'Asistencias', 'Pases_%', 'Regates', 'Recuperaciones', 'Duelos_Aereos', 'xG']

# Hay que normalizar porque si no los Pases (que son de 0-100) valen más que los Goles
escalador = MinMaxScaler()
df_norm = pd.DataFrame(escalador.fit_transform(df[cols_rendimiento]), columns=cols_rendimiento)

# --- BARRA LATERAL CON FILTROS ---
st.sidebar.header("Opciones de búsqueda")
lista_jugadores = df['Nombre'].tolist()
seleccionado = st.sidebar.selectbox("Elige un jugador para comparar:", lista_jugadores)

presupuesto = st.sidebar.slider("Presupuesto máximo (millones):", 0, 200, 150)
potencial_min = st.sidebar.slider("Potencial mínimo:", 70, 99, 85)

# --- ALGORITMO DE SIMILITUD (DISTANCIA EUCLIDIANA) ---
# Busco el vector del jugador que hemos elegido en el selectbox
idx_objetivo = df[df['Nombre'] == seleccionado].index[0]
vector_objetivo = df_norm.iloc[idx_objetivo].values

# Calculo la distancia de todos los jugadores contra mi objetivo
distancias = []
for i in range(len(df_norm)):
    vector_candidato = df_norm.iloc[i].values
    # Aplicamos la fórmula de la distancia euclidiana
    d = distance.euclidean(vector_objetivo, vector_candidato)
    distancias.append(d)

# Añado la columna de distancia al dataframe original para poder ordenar
df['Similitud'] = distancias

# --- MOSTRAR RESULTADOS ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"Perfil de {seleccionado}")
    # Gráfico de radar con Plotly
    valores = df.loc[idx_objetivo, cols_rendimiento].tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=valores,
        theta=cols_rendimiento,
        fill='toself',
        name=seleccionado
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
    st.plotly_chart(fig)

with col2:
    st.subheader("Jugadores más parecidos (Clones)")
    # Filtro por los criterios de la barra lateral y quito al propio jugador seleccionado
    filtros = (df['Valor_Mercado'] <= presupuesto) & (df['Potencial'] >= potencial_min) & (df['Nombre'] != seleccionado)
    recomendados = df[filtros].sort_values('Similitud').head(5)
    
    if recomendados.empty:
        st.warning("No hay jugadores que cumplan esos filtros de presupuesto/potencial.")
    else:
        st.table(recomendados[['Nombre', 'Equipo', 'Valor_Mercado', 'Similitud']])

st.divider()

# --- ESPECIALIZACIONES (Nivel Maestro) ---
st.header("Especializaciones Técnicas")

# 1. Agrupamiento por Roles (Clustering)
st.subheader("1. Agrupamiento Táctico con K-Means")
num_grupos = st.number_input("¿Cuántos grupos tácticos quieres crear?", 2, 5, 3)

modelo_km = KMeans(n_clusters=num_grupos, random_state=42)
df['Grupo_Tactico'] = modelo_km.fit_predict(df_norm)

st.write("Resultado del clustering (Segmentación de agentes):")
st.dataframe(df[['Nombre', 'Equipo', 'Grupo_Tactico']].sort_values('Grupo_Tactico'))

# 2. Algoritmo de Valor Justo (Gangas)
st.subheader("2. Detección de Activos Infravalorados")
# Calculo una puntuación de rendimiento sumando las métricas normalizadas
df['Suma_Rendimiento'] = df_norm.sum(axis=1)
# Si el rendimiento es alto y el precio es bajo, es una ganga
df['Relacion_Calidad_Precio'] = df['Suma_Rendimiento'] / (df['Valor_Mercado'] + 1)

gangas = df.sort_values('Relacion_Calidad_Precio', ascending=False).head(3)

st.info("Jugadores con mejor relación Rendimiento/Precio:")
for index, fila in gangas.iterrows():
    st.write(f"- **{fila['Nombre']}** ({fila['Equipo']}): Rendimiento alto para su coste de {fila['Valor_Mercado']}M.")