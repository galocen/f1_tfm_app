import streamlit as st
import pandas as pd
import numpy as np
import pickle
from app.data_loader import fetch_best_pract_laps, fetch_last_3_q
from app.data_processor import avg_last_3_positions, format_driver_times, get_driver_from_dummies, get_team_from_dummies
from app.data_predict import predict_laptime

st.set_page_config(page_title="Tandas en Prácticas", layout="wide", page_icon="📈")

st.title('Predicción de Tiempos de Clasificación')
st.markdown("Gonzalo Alocén Corral - Máster Big Data, Data Science & Inteligencia Artificial")

# Leemos el archivo del calendario y lo mostramos en pantalla
schedule = pd.read_csv('data/schedule.csv', sep=';', decimal=',', encoding='utf-8')
pred_schedule = schedule.drop(schedule[schedule['RoundNumber'] < 14].index)
pred_schedule.set_index('RoundNumber', inplace=True)
st.write(pred_schedule)

selected_round = st.selectbox("Selecciona carrera:", range(14, 26))
driver_times = fetch_best_pract_laps(selected_round, pred_schedule)

schedule.drop(schedule[schedule['RoundNumber'] == 0].index, inplace=True)
schedule.set_index('RoundNumber', inplace=True)

q_results = fetch_last_3_q(selected_round, schedule)
driver_times = avg_last_3_positions(driver_times, q_results)
driver_times = format_driver_times(driver_times, pred_schedule, selected_round)

try:
    predictions_Q1 = predict_laptime(driver_times, 'Q1')
except Exception as e:
    print(f"Error al hacer predicciones: {e}")
    st.stop()

drivers = get_driver_from_dummies(driver_times)
teams = get_team_from_dummies(driver_times)

# Crear DataFrame de resultados
results = pd.DataFrame({
    'Driver': drivers,
    'Team': teams,
    'Q1 Predicted Time': [f"{int(p//60)}:{p%60:06.3f}" for p in predictions_Q1],
    'Q1 Result Time': None
}).sort_values(by='Q1 Predicted Time').reset_index(drop=True)

# Mostrar y editar los resultados de Q1
st.subheader("Resultados Q1 - Puedes editar los tiempos reales")

edited_q1_results = st.data_editor(
    results, 
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
    key="q1_editor",
    column_config={
        "Q1 Result Time": st.column_config.TextColumn(
            "Q1 Result Time",
            help="Introduce el tiempo real de Q1 (formato: 1:23.456)"
        )
    }
)

# Botón para proceder a Q2
if st.button("Predecir Q2", key="predict_q2"):
    # Verificar que se han introducido tiempos reales
    if edited_q1_results['Q1 Result Time'].notna().any():
        st.success("Datos de Q1 actualizados!")
        
        # Procesar datos para Q2
        # Aquí puedes usar edited_q1_results para las predicciones de Q2
        st.write("Resultados Q1 finales:")
        st.write(edited_q1_results)
        
        # Continuar con la lógica de Q2...
        
    else:
        st.warning("Por favor, introduce al menos algunos tiempos reales de Q1 antes de continuar.")
