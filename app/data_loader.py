import streamlit as st
import fastf1 as ff1
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

BASE_URL = os.getenv("BASE_API_URL")

def fetch_data(endpoint, params=None):
    if params is None:
        params = {}

    url = f"{BASE_URL}{endpoint}"
    full_url = requests.Request('GET', url, params=params).prepare().url
    response = requests.get(full_url)
    response.raise_for_status()
    return pd.DataFrame(response.json())

@st.cache_data(show_spinner=False)
def fetch_sessions(meeting_key):
    # Devuelve las sesiones de un gran premio específico.
    df = fetch_data("sessions", {"meeting_key": meeting_key})

    # Combina el nombre de la sesión y la fecha de inicio para mostrarlo en pantalla
    df["label"] = df["session_name"] + " (" + df["date_start"] + ")"

    return df[["session_key", "label"]].drop_duplicates()

@st.cache_data(show_spinner=False)
def fetch_radios(session_key):
    # Devuelve los mensajes de radio para una sesión específica.
    return fetch_data("team_radio", {"session_key": session_key})

@st.cache_data(show_spinner=False)
def fetch_best_pract_laps(selected_round, schedule):
    if schedule['EventFormat'][selected_round] == 'sprint_qualifying' or schedule['EventFormat'][selected_round] == 'sprint' or schedule['EventFormat'][selected_round] == 'testing':
        st.warning(f"Carrera {selected_round} del 2025 no válida debido al formato: {schedule['EventFormat'][selected_round]}")
        st.stop()

    if schedule['Session1'][selected_round] != 'Practice 1' or schedule['Session2'][selected_round] != 'Practice 2' or schedule['Session3'][selected_round] != 'Practice 3':
        st.warning(f"Carrera {selected_round} del 2025 debido a la falta de sesiones de práctica.")
        st.stop()

    session_FP1 = ff1.get_session(2025, selected_round, 'FP1')
    session_FP2 = ff1.get_session(2025, selected_round, 'FP2')
    session_FP3 = ff1.get_session(2025, selected_round, 'FP3')

    session_FP1.load(laps=True, telemetry=False, weather=False, messages=False, livedata=None)
    session_FP2.load(laps=True, telemetry=False, weather=False, messages=False, livedata=None)
    session_FP3.load(laps=True, telemetry=False, weather=False, messages=False, livedata=None)

    try:
        laps_FP1 = session_FP1.laps.copy()
        laps_FP2 = session_FP2.laps.copy()
        laps_FP3 = session_FP3.laps.copy()
    except:
        st.warning("No hay datos de prácticas de esta sesión, no se puede estimar.")
        st.stop()

    laps_FP1['Session'] = 'FP1'
    laps_FP2['Session'] = 'FP2'
    laps_FP3['Session'] = 'FP3'
    all_laps = pd.concat([laps_FP1, laps_FP2, laps_FP3], ignore_index=True)

    # Obtener los 3 mejores tiempos por piloto
    best_laps = all_laps.sort_values(['Driver', 'LapTime'])
    top_3_laps = best_laps.groupby('Driver').head(3)
    
    # Devolvemos un DataFrame con los 3 mejores tiempos y el equipo
    driver_times = top_3_laps.groupby('Driver').agg({'LapTime': list, 'Team': 'first'})
    
    # Expandir las columnas de LapTime
    lap_times = pd.DataFrame(driver_times['LapTime'].tolist(), 
                             index=driver_times.index,
                             columns=['Best_Lap', 'Second_Best', 'Third_Best'])
    
    # Combinamos todo en un único DataFrame y reseteamos el índice
    driver_times = pd.concat([driver_times['Team'], lap_times], axis=1)
    driver_times = driver_times.reset_index()

    driver_times['Event'] = session_FP1.event.EventName
    driver_times['Year'] = 2025
    driver_times['RoundNumber'] = session_FP1.event.RoundNumber
    driver_times['Avg_Last_3_Positions'] = np.nan
    
    return driver_times

def fetch_last_3_q(selected_round, schedule):
    q_results = pd.DataFrame()
    i = 0
    
    while i < 3:
        selected_round -= 1
        if schedule['EventFormat'][selected_round] == 'sprint_qualifying' or schedule['Session1'][selected_round] != 'Practice 1' or schedule['Session2'][selected_round] != 'Practice 2' or schedule['Session3'][selected_round] != 'Practice 3':
            continue

        session_Q = ff1.get_session(2025, selected_round, 'Q')
        session_Q.load(laps=True, telemetry=False, weather=False, messages=False, livedata=None)
        results_Q = session_Q.results.copy()
        
        # Del mismo modo, creamos otra tabla con la posición final de cada piloto en la clasificación
        qualifying_result = pd.DataFrame(data={'Driver': results_Q['Abbreviation'], 'Position': results_Q['Position']})

        # Reseteamos el índice para que no sea el número de piloto
        qualifying_result.reset_index(drop=True, inplace=True)
        qualifying_result['RoundNumber'] = session_Q.event.RoundNumber
        qualifying_result['Year'] = 2025

        q_results = pd.concat([q_results, qualifying_result], ignore_index=True)
        i += 1
    
    return q_results

@st.cache_data(show_spinner=False)
def fetch_qualifying_results(selected_round, session):
    try:
        # Intentar obtener la sesión de clasificación
        session_Q = ff1.get_session(2025, selected_round, 'Q')
        
        # Verificar si la sesión existe y tiene datos
        try:
            session_Q.load(laps=True, telemetry=False, weather=False, messages=False, livedata=None)
        except Exception as load_error:
            st.info(f"Los datos de clasificación para la carrera {selected_round} aún no están disponibles")
            return {'times': {}, 'positions': {}}
        
        # Verificar si hay resultados disponibles
        if not hasattr(session_Q, 'results') or session_Q.results.empty:
            st.info(f"Los resultados de {session} para la carrera {selected_round} aún no están disponibles")
            return {'times': {}, 'positions': {}}
        
        # Obtener los resultados oficiales
        results = session_Q.results.copy()
        
        # Seleccionar la columna de tiempo apropiada según la sesión
        time_column = f'{session}'
        
        if time_column not in results.columns:
            
            # Intentar con nombres alternativos de columnas
            alternative_columns = [f'{session}_Time', f'{session.lower()}_time', f'{session.upper()}_TIME']
            found_column = None
            
            for alt_col in alternative_columns:
                if alt_col in results.columns:
                    found_column = alt_col
                    break
            
            if found_column:
                time_column = found_column
            else:
                st.warning(f"No hay datos de {session} disponibles para la carrera {selected_round}")
                return {'times': {}, 'positions': {}}
        
        # Crear diccionarios con los resultados
        q_times = {}
        q_positions = {}
        successful_mappings = 0
        
        # Filtrar y ordenar por tiempo para obtener posiciones reales
        valid_results = results[results[time_column].notna()].copy()
        
        if len(valid_results) > 0:
            # Convertir tiempos a segundos para ordenar correctamente
            def time_to_seconds(time_val):
                if pd.isna(time_val) or time_val == pd.NaT:
                    return float('inf')
                if hasattr(time_val, 'total_seconds'):
                    return time_val.total_seconds()
                return float('inf')
            
            valid_results['time_seconds'] = valid_results[time_column].apply(time_to_seconds)
            valid_results = valid_results.sort_values('time_seconds')
            
            # Asignar posiciones reales basadas en el orden de tiempos
            for position, (idx, row) in enumerate(valid_results.iterrows(), 1):
                try:
                    driver = row['Abbreviation'] if 'Abbreviation' in row else row['DriverNumber']
                    time_value = row[time_column]
                    
                    if pd.notna(time_value) and time_value != pd.NaT:
                        # Convertir el tiempo a formato string
                        if hasattr(time_value, 'total_seconds'):
                            total_seconds = time_value.total_seconds()
                            minutes = int(total_seconds // 60)
                            seconds = total_seconds % 60
                            formatted_time = f"{minutes}:{seconds:06.3f}"
                            q_times[driver] = formatted_time
                            q_positions[driver] = position
                            successful_mappings += 1
                        else:
                            pass  # Tipo de tiempo no reconocido
                    
                except Exception as row_error:
                    continue
        
        if successful_mappings > 0:
            st.success(f"✅ Cargados {successful_mappings} tiempos de {session}")
        else:
            st.info(f"ℹ️ No se encontraron tiempos válidos de {session}")
        
        return {'times': q_times, 'positions': q_positions}
    
    except Exception as e:
        st.warning(f"⚠️ Error al obtener resultados de {session} para la carrera {selected_round}: {e}")
        return {'times': {}, 'positions': {}}
