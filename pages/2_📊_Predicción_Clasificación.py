import streamlit as st
import pandas as pd
from app.data_loader import fetch_best_pract_laps, fetch_last_3_q, fetch_qualifying_results
from app.data_processor import avg_last_3_positions, format_driver_times, get_driver_from_dummies, get_team_from_dummies
from app.data_predict import predict_laptime, get_expected_features

# Configuración de la página Streamlit
st.set_page_config(page_title="Tandas en Prácticas", layout="wide", page_icon="📈")

st.title('Predicción de Tiempos de Clasificación')

# Panel informativo para el usuario sobre cómo usar la aplicación
st.info("""
### 📋 Cómo funciona esta aplicación:

**🎯 Objetivo**: Estimar la clasificación completa de F1 (Q1 → Q2 → Q3) paso a paso.

**📌 Instrucciones**:
1. **Selecciona una carrera** del calendario de F1 que aparece en la tabla.
2. **Q1**: Haz clic en "🏁 Hacer Predicciones Q1" → Se calcula y muestra la predicción de la Q1.
3. **Q2**: Una vez completada Q1, aparecerá el botón "🏁 Hacer Predicciones Q2" → Se calcula y muestra la predicción de la Q2.
4. **Q3**: Tras Q2, aparecerá "🏁 Hacer Predicciones Q3" → Se calcula y muestra la predicción de la Q3.

**🔄 Reset**: Se puede usar "🔄 Reset Predicciones" o cambiar el desplegable de selección de carrera para cambiar de carrera o reiniciar el proceso.
""")

st.markdown("---")

# Inicialización de variables de estado de sesión para mantener datos entre interacciones
# Estas variables controlan el flujo de la aplicación y almacenan resultados
if "predictions_made" not in st.session_state:
    st.session_state.predictions_made = False
if "q1_results" not in st.session_state:
    st.session_state.q1_results = None
if "q2_predictions_made" not in st.session_state:
    st.session_state.q2_predictions_made = False
if "q2_results" not in st.session_state:
    st.session_state.q2_results = None
if "q3_predictions_made" not in st.session_state:
    st.session_state.q3_predictions_made = False
if "q3_results" not in st.session_state:
    st.session_state.q3_results = None
if "driver_times_processed" not in st.session_state:
    st.session_state.driver_times_processed = None
if "current_round" not in st.session_state:
    st.session_state.current_round = None

# Cargar y mostrar el calendario de carreras disponibles para predicción
schedule = pd.read_csv('data/schedule.csv', sep=';', decimal=',', encoding='utf-8')
pred_schedule = schedule.drop(schedule[schedule['RoundNumber'] < 14].index)  # Solo carreras desde la ronda 14
pred_schedule.set_index('RoundNumber', inplace=True)
st.write(pred_schedule)

selected_round = st.selectbox("Selecciona carrera:", range(14, 26))

# Control de cambio de carrera: resetear estado completo si el usuario cambia de carrera
if selected_round != st.session_state.current_round:
    st.session_state.current_round = selected_round
    st.session_state.predictions_made = False
    st.session_state.q1_results = None
    st.session_state.q2_predictions_made = False
    st.session_state.q2_results = None
    st.session_state.q3_predictions_made = False
    st.session_state.q3_results = None
    st.session_state.driver_times_processed = None
    # Limpiar también los flags de carga de tiempos reales para forzar recarga
    if 'real_times_loaded' in st.session_state:
        del st.session_state.real_times_loaded
    if 'q2_real_times_loaded' in st.session_state:
        del st.session_state.q2_real_times_loaded
    if 'q3_real_times_loaded' in st.session_state:
        del st.session_state.q3_real_times_loaded

# SECCIÓN Q1: Botón principal para iniciar las predicciones de la primera sesión de clasificación
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_q1_button = st.button("🏁 Hacer Predicciones Q1", 
                                  key="predict_q1_btn", 
                                  width='stretch',
                                  type="primary")

# PROCESAMIENTO Q1: Lógica principal para obtener datos y generar predicciones
# Solo se ejecuta al pulsar el botón o si ya se han hecho predicciones anteriormente
if predict_q1_button or st.session_state.predictions_made:
    if not st.session_state.predictions_made:
        try:
            # 1. Obtener tiempos de prácticas libres para usar como base de predicción
            driver_times = fetch_best_pract_laps(selected_round, pred_schedule)

            # 2. Obtener resultados de clasificaciones previas para mejorar predicciones
            # Necesitamos el schedule completo (no filtrado) para esta función
            full_schedule = pd.read_csv('data/schedule.csv', sep=';', decimal=',', encoding='utf-8')
            q_results = fetch_last_3_q(selected_round, full_schedule)
            driver_times = avg_last_3_positions(driver_times, q_results)
            driver_times = format_driver_times(driver_times, pred_schedule, selected_round)

            # 3. Generar predicciones usando el modelo entrenado Q1
            predictions_Q1 = predict_laptime(driver_times, 'Q1')

            # 4. Extraer información de pilotos y equipos de los datos procesados
            drivers = get_driver_from_dummies(driver_times)
            teams = get_team_from_dummies(driver_times)

            # 5. Almacenar datos procesados para reutilizar en Q2 y Q3
            st.session_state.driver_times_processed = driver_times.copy()

            # 6. Crear DataFrame con resultados organizados y formateados para visualización
            results_df = pd.DataFrame({
                'Driver': drivers,
                'Team': teams,
                'Q1 Predicted Time': [f"{int(p//60)}:{p%60:06.3f}" for p in predictions_Q1],
                'Q1 Result Time': None,     # Se completará con tiempos reales si están disponibles
                'Predicted Position': None,
                'Real Position': None       # Se completará con posiciones reales si están disponibles
            }).sort_values(by='Q1 Predicted Time').reset_index(drop=True)
            
            # 7. Asignar de las posiciones predichas
            results_df['Predicted Position'] = range(1, len(results_df) + 1)
            
            st.session_state.q1_results = results_df
            st.session_state.predictions_made = True
            
        except Exception as e:
            st.error(f"Error al hacer predicciones: {str(e)}")
            import traceback
            st.error(f"Detalles del error: {traceback.format_exc()}")
            st.stop()

    # VISUALIZACIÓN Q1: Mostrar tabla de resultados con predicciones y datos reales (si disponibles)
    if st.session_state.predictions_made:
        st.subheader("Resultados Q1")
        
        # INTEGRACIÓN DATOS REALES Q1: Cargar y combinar resultados reales con predicciones
        # Solo se ejecuta una vez por sesión para optimizar rendimiento
        if 'real_times_loaded' not in st.session_state:
            try:
                # Intentar obtener los resultados oficiales de Q1 de la API FastF1
                q1_real_data = fetch_qualifying_results(selected_round, 'Q1')
                
                if q1_real_data and q1_real_data.get('times'):
                    q1_real_results = q1_real_data['times']
                    q1_real_positions = q1_real_data['positions']
                    
                    # ALGORITMO DE MATCHING: Asociar pilotos de predicción con resultados reales
                    for idx, row in st.session_state.q1_results.iterrows():
                        driver = row['Driver']
                        # Buscar coincidencia exacta (más confiable)
                        matched = False
                        for real_driver, real_time in q1_real_results.items():
                            if driver == real_driver:
                                st.session_state.q1_results.loc[idx, 'Q1 Result Time'] = real_time
                                st.session_state.q1_results.loc[idx, 'Real Position'] = q1_real_positions.get(real_driver, None)
                                matched = True
                                break
                    
                    # REORDENAMIENTO POR DATOS REALES: Priorizar orden oficial cuando esté disponible
                    if 'Real Position' in st.session_state.q1_results.columns and st.session_state.q1_results['Real Position'].notna().any():
                        # Ordenar por posición real oficial, manteniendo predicciones al final si no hay datos reales
                        st.session_state.q1_results = st.session_state.q1_results.sort_values(
                            by='Real Position', 
                            na_position='last'
                        ).reset_index(drop=True)
                else:
                    pass  # Datos oficiales aún no disponibles - mantener solo predicciones
                    
            except Exception as e:
                pass  # Falló la conexión con API - continuar solo con predicciones
            
            # Marcar que ya se intentó cargar datos reales (evitar llamadas repetidas)
            st.session_state.real_times_loaded = True
        
        # TABLA DE RESULTADOS Q1: Visualización unificada de predicciones vs realidad
        st.dataframe(
            st.session_state.q1_results,
            hide_index=True,
            width='stretch',
            column_config={
                "Driver": st.column_config.TextColumn("Piloto", width="medium"),
                "Team": st.column_config.TextColumn("Equipo", width="medium"),
                "Predicted Position": st.column_config.NumberColumn("Pos. Predicha", width="small"),
                "Real Position": st.column_config.NumberColumn("Pos. Real", width="small"),
                "Q1 Predicted Time": st.column_config.TextColumn("Tiempo Predicho", width="medium"),
                "Q1 Result Time": st.column_config.TextColumn("Tiempo Real", width="medium")
            }
        )
        
        # CONTROLES DE NAVEGACIÓN: Botones para reset y continuar a siguiente fase
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Reset Predicciones", key="reset_predictions"):
                # RESET COMPLETO: Volver al estado inicial limpiando todas las sesiones
                st.session_state.predictions_made = False
                st.session_state.q1_results = None
                st.session_state.q2_predictions_made = False
                st.session_state.q2_results = None
                st.session_state.q3_predictions_made = False
                st.session_state.q3_results = None
                st.session_state.driver_times_processed = None
                # Limpiar flags de carga de datos reales para forzar nueva carga
                if 'real_times_loaded' in st.session_state:
                    del st.session_state.real_times_loaded
                if 'q2_real_times_loaded' in st.session_state:
                    del st.session_state.q2_real_times_loaded
                if 'q3_real_times_loaded' in st.session_state:
                    del st.session_state.q3_real_times_loaded
                st.rerun()
        
        with col2:
            if st.button("🏁 Hacer Predicciones Q2", 
                         key="predict_q2", 
                         width='stretch',
                         type="primary"):
                with st.spinner("Procesando datos y haciendo predicciones Q2..."):
                    try:
                        # PROCESAMIENTO Q2: Aplicar reglas F1 (solo top 15 de Q1 avanzan a Q2)
                        if st.session_state.driver_times_processed is not None:
                            # FILTRADO DE PILOTOS: Seleccionar top 15 basado en posiciones reales o predichas
                            if 'Real Position' in st.session_state.q1_results.columns:
                                # Usar posiciones oficiales de Q1 si están disponibles
                                top15_q1 = st.session_state.q1_results[
                                    (st.session_state.q1_results['Real Position'] >= 1) & 
                                    (st.session_state.q1_results['Real Position'] <= 15)
                                ].copy()
                            else:
                                # Si no están disponibles, usar predicciones Q1 como criterio (primeros 15)
                                top15_q1 = st.session_state.q1_results.head(15).copy()
                            
                            # MAPEO DE PILOTOS CLASIFICADOS: Lista de nombres que avanzan a Q2
                            qualified_drivers = top15_q1['Driver'].tolist()
                            
                            # RECUPERACIÓN DE DATOS BASE: Extraer pilotos de datos procesados originales
                            all_drivers = get_driver_from_dummies(st.session_state.driver_times_processed)
                            
                            # MATCHING DE ÍNDICES: Encontrar posiciones de pilotos clasificados
                            qualified_indices = []
                            for i, driver in enumerate(all_drivers):
                                if driver in qualified_drivers:
                                    qualified_indices.append(i)
                            
                            # FILTRADO FINAL: Solo datos de pilotos que avanzan a Q2
                            driver_times_q2 = st.session_state.driver_times_processed.iloc[qualified_indices].copy()
                            
                            # VALIDACIÓN DEL MODELO: Verificar características requeridas por modelo Q2
                            expected_columns_q2 = get_expected_features('Q2')
                            
                            if expected_columns_q2 is None:
                                st.error("No se pudieron obtener las características esperadas del modelo Q2")
                            else:
                                # PREPARACIÓN Q1 COMO FEATURE: Convertir tiempos Q1 a formato numérico
                                q1_predicted_times = []
                                for idx, row in top15_q1.iterrows():
                                    q1_time = row['Q1 Predicted Time']
                                    try:
                                        # Conversión de formato "m:ss.fff" a segundos totales
                                        if ':' in str(q1_time):
                                            parts = str(q1_time).split(':')
                                            minutes = int(parts[0])
                                            seconds = float(parts[1])
                                            total_seconds = minutes * 60 + seconds
                                            q1_predicted_times.append(total_seconds)
                                        else:
                                            q1_predicted_times.append(float(q1_time))
                                    except:
                                        q1_predicted_times.append(80.0)  # Valor por defecto
                            
                            # Añadir la columna Q1
                            driver_times_q2['Q1'] = q1_predicted_times
                            
                            # Si hay tiempos reales de Q1, reemplazar las predicciones (solo primeros 15)
                            if top15_q1['Q1 Result Time'].notna().any():
                                q1_real_times = []
                                for i, (idx, row) in enumerate(top15_q1.iterrows()):
                                    q1_time = row['Q1 Result Time']
                                    if pd.notna(q1_time) and q1_time is not None:
                                        try:
                                            # Convertir de formato "m:ss.fff" a segundos
                                            if ':' in str(q1_time):
                                                parts = str(q1_time).split(':')
                                                minutes = int(parts[0])
                                                seconds = float(parts[1])
                                                total_seconds = minutes * 60 + seconds
                                                q1_real_times.append(total_seconds)
                                            else:
                                                q1_real_times.append(float(q1_time))
                                        except:
                                            # Si falla la conversión, mantener el valor predicho
                                            q1_real_times.append(q1_predicted_times[i])
                                    else:
                                        # Si no hay tiempo real, mantener el valor predicho
                                        q1_real_times.append(q1_predicted_times[i])
                                
                                # Reemplazar con tiempos reales donde sea posible
                                driver_times_q2['Q1'] = q1_real_times
                            
                            # Reorganizar las columnas en el orden exacto que espera el modelo
                            missing_cols = set(expected_columns_q2) - set(driver_times_q2.columns)
                            if missing_cols:
                                for col in missing_cols:
                                    driver_times_q2[col] = 0
                            
                            # Reordenar según las características esperadas por el modelo
                            driver_times_q2_ordered = driver_times_q2[expected_columns_q2]
                            
                            # Predecir Q2 - esto debería funcionar ahora con las características correctas
                            predictions_Q2 = predict_laptime(driver_times_q2_ordered, 'Q2')

                            drivers = get_driver_from_dummies(driver_times_q2_ordered)
                            teams = get_team_from_dummies(driver_times_q2_ordered)

                            # Crear DataFrame de resultados Q2 con posiciones
                            results_df_q2 = pd.DataFrame({
                                'Driver': drivers,
                                'Team': teams,
                                'Q2 Predicted Time': [f"{int(p//60)}:{p%60:06.3f}" for p in predictions_Q2],
                                'Q2 Result Time': None,
                                'Predicted Position': None,
                                'Real Position': None
                            }).sort_values(by='Q2 Predicted Time').reset_index(drop=True)
                            
                            # Añadir posiciones predichas (1, 2, 3, ...)
                            results_df_q2['Predicted Position'] = range(1, len(results_df_q2) + 1)
                            
                            st.session_state.q2_results = results_df_q2

                            st.session_state.q2_predictions_made = True
                        else:
                            st.error("Error: Necesitas hacer primero las predicciones de Q1")
                        
                    except Exception as e:
                        st.error(f"Error al hacer predicciones Q2: {e}")
                        st.stop()

# VISUALIZACIÓN Q2: Mostrar resultados de la segunda sesión de clasificación
if st.session_state.q2_predictions_made and st.session_state.q2_results is not None:
    st.markdown("---")
    st.subheader("Resultados Q2")
    
    # Cargar resultados reales de Q2 si están disponibles
    if 'q2_real_times_loaded' not in st.session_state:
        try:
            # Intentar obtener los resultados reales de Q2
            q2_real_data = fetch_qualifying_results(selected_round, 'Q2')
            
            if q2_real_data and q2_real_data.get('times'):
                q2_real_results = q2_real_data['times']
                q2_real_positions = q2_real_data['positions']
                
                # Mapear los resultados reales a los pilotos en la tabla
                for idx, row in st.session_state.q2_results.iterrows():
                    driver = row['Driver']
                    # Buscar el piloto en los resultados reales (puede estar con abreviación)
                    for real_driver, real_time in q2_real_results.items():
                        if driver in real_driver or real_driver in driver:
                            st.session_state.q2_results.loc[idx, 'Q2 Result Time'] = real_time
                            st.session_state.q2_results.loc[idx, 'Real Position'] = q2_real_positions.get(real_driver, None)
                            break
                
                # Reordenar por posición real si está disponible
                if 'Real Position' in st.session_state.q2_results.columns and st.session_state.q2_results['Real Position'].notna().any():
                    st.session_state.q2_results = st.session_state.q2_results.sort_values(
                        by='Real Position', 
                        na_position='last'
                    ).reset_index(drop=True)
            else:
                pass  # Resultados reales no disponibles aún
                
        except Exception as e:
            pass  # Error al cargar resultados reales
        
        st.session_state.q2_real_times_loaded = True
    
    # Mostrar tabla única con predicciones y resultados reales de Q2
    st.dataframe(
        st.session_state.q2_results,
        hide_index=True,
        width='stretch',
        column_config={
            "Driver": st.column_config.TextColumn("Piloto", width="medium"),
            "Team": st.column_config.TextColumn("Equipo", width="medium"),
            "Predicted Position": st.column_config.NumberColumn("Pos. Predicha", width="small"),
            "Real Position": st.column_config.NumberColumn("Pos. Real", width="small"),
            "Q2 Predicted Time": st.column_config.TextColumn("Tiempo Predicho Q2", width="medium"),
            "Q2 Result Time": st.column_config.TextColumn("Tiempo Real Q2", width="medium")
        }
    )
    
    # Botones adicionales para Q2
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reset Predicciones", key="reset_q2_predictions"):
            # Reset completo al estado inicial
            st.session_state.predictions_made = False
            st.session_state.q1_results = None
            st.session_state.q2_predictions_made = False
            st.session_state.q2_results = None
            st.session_state.q3_predictions_made = False
            st.session_state.q3_results = None
            st.session_state.driver_times_processed = None
            # Eliminar cualquier estado de tiempos reales cargados
            if 'real_times_loaded' in st.session_state:
                del st.session_state.real_times_loaded
            if 'q2_real_times_loaded' in st.session_state:
                del st.session_state.q2_real_times_loaded
            if 'q3_real_times_loaded' in st.session_state:
                del st.session_state.q3_real_times_loaded
            st.rerun()
    
    with col2:
        if st.button("🏁 Hacer Predicciones Q3", 
                     key="predict_q3", 
                     width='stretch',
                     type="primary"):
            with st.spinner("Procesando datos y haciendo predicciones Q3..."):
                try:
                    # Verificar que se hicieron las predicciones Q2 primero
                    if st.session_state.q2_predictions_made and st.session_state.q2_results is not None:
                        # Filtrar pilotos con posición real 1-10 de Q2 que pasan a Q3
                        if 'Real Position' in st.session_state.q2_results.columns:
                            # Si hay posiciones reales en Q2, usar esas para filtrar
                            top10_q2 = st.session_state.q2_results[
                                (st.session_state.q2_results['Real Position'] >= 1) & 
                                (st.session_state.q2_results['Real Position'] <= 10)
                            ].copy()
                        else:
                            # Si no hay posiciones reales, usar las predichas (primeros 10)
                            top10_q2 = st.session_state.q2_results.head(10).copy()
                        
                        # También necesitamos Q1 con posiciones reales para referencias
                        if 'Real Position' in st.session_state.q1_results.columns:
                            top15_q1 = st.session_state.q1_results[
                                (st.session_state.q1_results['Real Position'] >= 1) & 
                                (st.session_state.q1_results['Real Position'] <= 15)
                            ].copy()
                        else:
                            top15_q1 = st.session_state.q1_results.head(15).copy()
                        
                        # Obtener los pilotos que clasificaron para Q3 (por nombre, no por índice)
                        qualified_q3_drivers = top10_q2['Driver'].tolist()
                        
                        # Obtener los pilotos de driver_times_processed para hacer el matching
                        from app.data_processor import get_driver_from_dummies
                        all_drivers = get_driver_from_dummies(st.session_state.driver_times_processed)
                        
                        # Encontrar los índices de los pilotos clasificados para Q3 en driver_times_processed
                        qualified_q3_indices = []
                        for i, driver in enumerate(all_drivers):
                            if driver in qualified_q3_drivers:
                                qualified_q3_indices.append(i)
                        
                        # Filtrar driver_times_processed para solo incluir los pilotos clasificados para Q3
                        driver_times_q3 = st.session_state.driver_times_processed.iloc[qualified_q3_indices].copy()
                        
                        # Obtener las características esperadas directamente del modelo Q3
                        expected_columns_q3 = get_expected_features('Q3')
                        
                        if expected_columns_q3 is None:
                            st.error("No se pudieron obtener las características esperadas del modelo Q3")
                        else:
                            # Añadir columna Q1 con las predicciones/tiempos reales (solo primeros 10)
                            q1_times = []
                            for i, (idx, row) in enumerate(top10_q2.iterrows()):
                                # Buscar el piloto correspondiente en Q1
                                driver = top10_q2.iloc[i]['Driver']
                                q1_row = top15_q1[top15_q1['Driver'] == driver].iloc[0]
                                q1_time = q1_row['Q1 Result Time'] if pd.notna(q1_row['Q1 Result Time']) else q1_row['Q1 Predicted Time']
                                try:
                                    if ':' in str(q1_time):
                                        parts = str(q1_time).split(':')
                                        minutes = int(parts[0])
                                        seconds = float(parts[1])
                                        total_seconds = minutes * 60 + seconds
                                        q1_times.append(total_seconds)
                                    else:
                                        q1_times.append(float(q1_time))
                                except:
                                    q1_times.append(80.0)
                            
                            driver_times_q3['Q1'] = q1_times
                            
                            # Añadir columna Q2 con las predicciones/tiempos reales (solo primeros 10)
                            q2_times = []
                            for idx, row in top10_q2.iterrows():
                                q2_time = row['Q2 Result Time'] if pd.notna(row['Q2 Result Time']) else row['Q2 Predicted Time']
                                try:
                                    if ':' in str(q2_time):
                                        parts = str(q2_time).split(':')
                                        minutes = int(parts[0])
                                        seconds = float(parts[1])
                                        total_seconds = minutes * 60 + seconds
                                        q2_times.append(total_seconds)
                                    else:
                                        q2_times.append(float(q2_time))
                                except:
                                    q2_times.append(79.0)
                            
                            driver_times_q3['Q2'] = q2_times
                            
                            # Reordenar según las características esperadas por el modelo
                            driver_times_q3_ordered = driver_times_q3[expected_columns_q3]
                            
                            # Predecir Q3
                            predictions_Q3 = predict_laptime(driver_times_q3_ordered, 'Q3')
                            
                            if predictions_Q3 is not None:
                                drivers = get_driver_from_dummies(driver_times_q3_ordered)
                                teams = get_team_from_dummies(driver_times_q3_ordered)

                                # Crear DataFrame de resultados Q3 con posiciones
                                results_df_q3 = pd.DataFrame({
                                    'Driver': drivers,
                                    'Team': teams,
                                    'Q3 Predicted Time': [f"{int(p//60)}:{p%60:06.3f}" for p in predictions_Q3],
                                    'Q3 Result Time': None,
                                    'Predicted Position': None,
                                    'Real Position': None
                                }).sort_values(by='Q3 Predicted Time').reset_index(drop=True)
                                
                                # Añadir posiciones predichas (1, 2, 3, ...)
                                results_df_q3['Predicted Position'] = range(1, len(results_df_q3) + 1)
                                
                                st.session_state.q3_results = results_df_q3

                                st.session_state.q3_predictions_made = True
                            else:
                                st.error("Error en la predicción Q3")
                    else:
                        st.error("Error: Necesitas hacer primero las predicciones de Q1 y Q2")
                    
                except Exception as e:
                    st.error(f"Error al hacer predicciones Q3: {e}")

# VISUALIZACIÓN Q3: Mostrar resultados finales de la pole position (top 10)
if st.session_state.q3_predictions_made and st.session_state.q3_results is not None:
    st.markdown("---")
    st.subheader("🏁 Resultados Q3")
    
    # Cargar resultados reales de Q3 si están disponibles
    if 'q3_real_times_loaded' not in st.session_state:
        try:
            # Intentar obtener los resultados reales de Q3
            q3_real_data = fetch_qualifying_results(selected_round, 'Q3')
            
            if q3_real_data and q3_real_data.get('times'):
                q3_real_results = q3_real_data['times']
                q3_real_positions = q3_real_data['positions']
                
                # Mapear los resultados reales a los pilotos en la tabla
                for idx, row in st.session_state.q3_results.iterrows():
                    driver = row['Driver']
                    # Buscar el piloto en los resultados reales (puede estar con abreviación)
                    for real_driver, real_time in q3_real_results.items():
                        if driver in real_driver or real_driver in driver:
                            st.session_state.q3_results.loc[idx, 'Q3 Result Time'] = real_time
                            st.session_state.q3_results.loc[idx, 'Real Position'] = q3_real_positions.get(real_driver, None)
                            break
                
                # Reordenar por posición real si está disponible
                if 'Real Position' in st.session_state.q3_results.columns and st.session_state.q3_results['Real Position'].notna().any():
                    st.session_state.q3_results = st.session_state.q3_results.sort_values(
                        by='Real Position', 
                        na_position='last'
                    ).reset_index(drop=True)
            else:
                pass  # Resultados reales no disponibles aún
                
        except Exception as e:
            pass  # Error al cargar resultados reales
        
        st.session_state.q3_real_times_loaded = True
    
    # Mostrar tabla única con predicciones y resultados reales de Q3
    st.dataframe(
        st.session_state.q3_results,
        hide_index=True,
        width='stretch',
        column_config={
            "Driver": st.column_config.TextColumn("Piloto", width="medium"),
            "Team": st.column_config.TextColumn("Equipo", width="medium"),
            "Predicted Position": st.column_config.NumberColumn("Pos. Predicha", width="small"),
            "Real Position": st.column_config.NumberColumn("Pos. Real", width="small"),
            "Q3 Predicted Time": st.column_config.TextColumn("Tiempo Predicho Q3", width="medium"),
            "Q3 Result Time": st.column_config.TextColumn("Tiempo Real Q3", width="medium")
        }
    )
    
    # Opciones adicionales para Q3
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reset Predicciones", key="reset_q3"):
            # Reset completo al estado inicial
            st.session_state.predictions_made = False
            st.session_state.q1_results = None
            st.session_state.q2_predictions_made = False
            st.session_state.q2_results = None
            st.session_state.q3_predictions_made = False
            st.session_state.q3_results = None
            st.session_state.driver_times_processed = None
            # Eliminar cualquier estado de tiempos reales cargados
            if 'real_times_loaded' in st.session_state:
                del st.session_state.real_times_loaded
            if 'q2_real_times_loaded' in st.session_state:
                del st.session_state.q2_real_times_loaded
            if 'q3_real_times_loaded' in st.session_state:
                del st.session_state.q3_real_times_loaded
            st.rerun()
