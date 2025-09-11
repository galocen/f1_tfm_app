import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
import tempfile
import os
import librosa

def avg_last_3_positions(driver_times, q_results):
    # Recorremos cada fila de driver_times para calcular la media de las posiciones de clasificación de las últimas 3 rondas
    for index, row in driver_times.iterrows():
        # Obtenemos el número de ronda y piloto de la fila actual
        round_number = row['RoundNumber']
        driver = row['Driver']
        
        # Obtenemos las posiciones de clasificación de las últimas 3 rondas para el piloto
        previous_results = []

        # Filtrar resultados del piloto específico en q_results
        driver_q_results = q_results[q_results['Driver'] == driver]
        
        # Iterar sobre cada resultado de clasificación del piloto
        for _, prev_round in driver_q_results.iterrows():
            result = prev_round['Position']
            if pd.notna(result):  # Verificar que no sea NaN
                previous_results.append(result)
        
        # Calculamos la media de las posiciones si hay resultados
        if previous_results:
            driver_times.at[index, 'Avg_Last_3_Positions'] = np.mean(previous_results)
        else:
            driver_times.at[index, 'Avg_Last_3_Positions'] = np.nan

    driver_times.dropna(subset=['Avg_Last_3_Positions'], inplace=True)
    driver_times.reset_index(drop=True, inplace=True)
    driver_times.drop(columns=['RoundNumber'], inplace=True)
    
    return driver_times

def format_driver_times(driver_times, pred_schedule, selected_round):
    event_list = ['Event_Australian Grand Prix', 'Event_Austrian Grand Prix',
                'Event_Azerbaijan Grand Prix', 'Event_Bahrain Grand Prix',
                'Event_Belgian Grand Prix', 'Event_Brazilian Grand Prix',
                'Event_British Grand Prix', 'Event_Canadian Grand Prix',
                'Event_Chinese Grand Prix', 'Event_Dutch Grand Prix',
                'Event_Emilia Romagna Grand Prix', 'Event_Hungarian Grand Prix',
                'Event_Italian Grand Prix', 'Event_Japanese Grand Prix',
                'Event_Las Vegas Grand Prix', 'Event_Mexican Grand Prix',
                'Event_Miami Grand Prix', 'Event_Monaco Grand Prix', 'Event_Others',
                'Event_Qatar Grand Prix', 'Event_Saudi Arabian Grand Prix',
                'Event_Singapore Grand Prix', 'Event_Spanish Grand Prix',
                'Event_United States Grand Prix', 'Driver_OTH']
    
    sorted_columns = ['Best_Lap', 'Second_Best', 'Third_Best', 'Year', 'Avg_Last_3_Positions',
        'Driver_ALO', 'Driver_ANT', 'Driver_BEA', 'Driver_BOR', 'Driver_COL',
        'Driver_GAS', 'Driver_HAD', 'Driver_HAM', 'Driver_HUL', 'Driver_LAW',
        'Driver_LEC', 'Driver_NOR', 'Driver_OCO', 'Driver_OTH', 'Driver_PIA',
        'Driver_RUS', 'Driver_SAI', 'Driver_STR', 'Driver_TSU', 'Driver_VER',
        'Team_Aston Martin', 'Team_Ferrari', 'Team_Haas F1 Team',
        'Team_Kick Sauber', 'Team_McLaren', 'Team_Mercedes', 'Team_Racing Bulls',
        'Team_Red Bull Racing', 'Team_Williams', 'Event_Australian Grand Prix',
        'Event_Austrian Grand Prix', 'Event_Azerbaijan Grand Prix',
        'Event_Bahrain Grand Prix', 'Event_Belgian Grand Prix', 'Event_Brazilian Grand Prix',
        'Event_British Grand Prix', 'Event_Canadian Grand Prix', 'Event_Chinese Grand Prix',
        'Event_Dutch Grand Prix', 'Event_Emilia Romagna Grand Prix', 'Event_Hungarian Grand Prix',
        'Event_Italian Grand Prix', 'Event_Japanese Grand Prix', 'Event_Las Vegas Grand Prix',
        'Event_Mexican Grand Prix', 'Event_Miami Grand Prix', 'Event_Monaco Grand Prix',
        'Event_Others','Event_Qatar Grand Prix', 'Event_Saudi Arabian Grand Prix',
        'Event_Singapore Grand Prix', 'Event_Spanish Grand Prix', 'Event_United States Grand Prix']

    driver_times = pd.get_dummies(driver_times, columns=['Driver', 'Team'], drop_first=True)
    
    for event in event_list:
        if event == f'Event_{pred_schedule["EventName"][selected_round]}':
            driver_times[event] = True
        else:
            driver_times[event] = False

    driver_times.drop(columns=['Event'], inplace=True)

    time_columns = ['Best_Lap', 'Second_Best', 'Third_Best']

    for col in time_columns:
        if col in driver_times.columns:
            driver_times[col] = pd.to_timedelta(driver_times[col]).dt.total_seconds()

    return driver_times[sorted_columns]

def get_driver_from_dummies(df_con_dummies):
    """
    Reconstruye la información del driver desde las variables dummy
    """
    drivers = []
    
    for index, row in df_con_dummies.iterrows():
        # Buscar qué columna de driver tiene valor True/1
        driver_cols = [col for col in df_con_dummies.columns if col.startswith('Driver_')]
        
        driver = None
        for col in driver_cols:
            if row[col] == True or row[col] == 1:
                driver = col.replace('Driver_', '')
                break
        
        # Si no se encontró ningún driver en las dummies, es el que se eliminó en drop_first=True
        if driver is None:
            driver = "ALB"  # El primer driver alfabéticamente que se eliminó
            
        drivers.append(driver)
    
    return drivers

def get_team_from_dummies(df_con_dummies):
    """
    Reconstruye la información del team desde las variables dummy
    """
    teams = []
    
    for index, row in df_con_dummies.iterrows():
        team_cols = [col for col in df_con_dummies.columns if col.startswith('Team_')]
        
        team = None
        for col in team_cols:
            if row[col] == True or row[col] == 1:
                team = col.replace('Team_', '')
                break
        
        # Si no se encontró ningún team, es el que se eliminó en drop_first=True
        if team is None:
            team = "Alpine"  # El primer team alfabéticamente que se eliminó
            
        teams.append(team)
    
    return teams

def download_and_process_audio(url):
    """Descarga y procesa el audio en memoria usando librosa"""
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            # Escribir temporalmente a disco solo para librosa
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_file.flush()  # Asegurar que se escriba
                
                # Cargar con librosa
                audio, sr = librosa.load(tmp_file.name, sr=16000)
            
            # Limpiar archivo temporal
            try:
                os.unlink(tmp_file.name)
            except (PermissionError, FileNotFoundError):
                pass
                
            return audio
        return None
    except Exception as e:
        st.error(f"Error al procesar audio: {str(e)}")
        return None
