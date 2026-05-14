import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.spatial import distance
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE LA CONSOLA ---
st.set_page_config(page_title="Centro de Inteligencia de la Alianza", layout="wide", page_icon="🥷")

# Estilo visual inspirado en la Alianza Shinobi
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

st.title("🥷 Centro de Inteligencia de la Alianza")
st.subheader("Sistema de Rastreo y Clonación Estadística")

# --- CARGA DE DATOS (SISTEMA DE CACHÉ) ---
@st.cache_data
def cargar_datos():
    df = pd.read_csv('players_data.csv')
    return df

try:
    df = cargar_datos()
except FileNotFoundError:
    st.error("⚠️ Error: No se encontró el archivo 'players_data.csv'. Asegúrate de que esté en la misma carpeta.")
    st.stop()

# --- PREPROCESAMIENTO ---
# Columnas para el análisis de rendimiento
metricas_cols = ['Goles', 'Asistencias', 'Pases_%', 'Regates', 'Recuperaciones', 'Duelos_Aereos', 'xG']

# Normalización (0 a 1) para evitar sesgos en la Distancia Euclidiana
scaler = MinMaxScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df[metricas_cols]), columns=metricas_cols)

# --- PANEL LATERAL (UX) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/188/188837.png", width=100)
st.sidebar.header("Parámetros de Búsqueda")

target_player = st.sidebar.selectbox("Seleccionar Activo Objetivo", df['Nombre'].unique())
max_price = st.sidebar.slider("Presupuesto Máximo (M€)", 0, int(df['Valor_Mercado'].max()), 200)
min_potencial = st.sidebar.slider("Potencial Mínimo", int(df['Potencial'].min()), 100, 80)

# Filtrado por presupuesto y potencial
df_filtrado = df[(df['Valor_Mercado'] <= max_price) & (df['Potencial'] >= min_potencial)].copy()

# --- ALGORITMO DE SIMILITUD (DISTANCIA EUCLIDIANA) ---
def calcular_similitud(nombre_obj, df_base, df_normalizado):
    idx_obj = df_base[df_base['Nombre'] == nombre_obj].index[0]
    vector_obj = df_normalizado.iloc[idx_obj].values
    
    distancias = []
    for i in range(len(df_normalizado)):
        vector_candidato = df_normalizado.iloc[i].values
        dist = distance.euclidean(vector_obj, vector_candidato)
        distancias.append(dist)
    
    return distancias

df['Distancia'] = calcular_similitud(target_player, df, df_norm)

# --- VISUALIZACIÓN ---
col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("### 📊 Perfil Táctico")
    # Radar Chart
    player_data = df[df['Nombre'] == target_player][metricas_cols].iloc[0].tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=player_data,
        theta=metricas_cols,
        fill='toself',
        line_color='#e63946',
        name=target_player
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, df[metricas_cols].max().max()])), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 👥 Clones Estadísticos Identificados")
    # Mostrar los 5 más cercanos (excluyendo al seleccionado si es posible)
    recomendaciones = df[df['Nombre'] != target_player].sort_values('Distancia').head(5)
    st.table(recomendaciones[['Nombre', 'Equipo', 'Valor_Mercado', 'Potencial']])

st.divider()

# --- ESPECIALIZACIONES DE MAESTRO ---
st.header("⚡ Módulos Avanzados de Inteligencia")

tab1, tab2 = st.tabs(["🎯 Agrupamiento Táctico", "💰 Estrategia de Valor Justo"])

with tab1:
    st.subheader("Segmentación por Roles (Clustering)")
    n_clusters = st.select_slider("Ajustar precisión de roles", options=[2, 3, 4, 5], value=3)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Rol_ID'] = kmeans.fit_predict(df_norm)
    
    st.write("El algoritmo ha identificado los siguientes grupos tácticos:")
    st.dataframe(df[['Nombre', 'Equipo', 'Rol_ID']].sort_values('Rol_ID'), use_container_width=True)

with tab2:
    st.subheader("Análisis de Activos Infravalorados")
    # Lógica: Rendimiento (suma de métricas norm) / Valor de Mercado
    df['Rendimiento_Puro'] = df_norm.sum(axis=1)
    df['Indice_Ganga'] = df['Rendimiento_Puro'] / (df['Valor_Mercado'] + 1) # +1 para evitar división por cero
    
    gangas = df.sort_values('Indice_Ganga', ascending=False).head(5)
    
    st.success("Los siguientes activos tienen un rendimiento superior a su precio de mercado actual:")
    for i, row in gangas.iterrows():
        st.write(f"🔹 **{row['Nombre']}** ({row['Equipo']}) - Valor: {row['Valor_Mercado']}M€")