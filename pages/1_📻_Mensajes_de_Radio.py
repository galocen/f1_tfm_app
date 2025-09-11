import streamlit as st
from transformers import pipeline
import torch
from app.data_loader import (
    fetch_data,
    fetch_sessions,
    fetch_radios
)
from app.data_processor import download_and_process_audio

# Configuración de la página Streamlit
st.set_page_config(page_title="Mensajes de Radio", layout="wide", page_icon="📻")

st.title('Mensajes de Radio')

# Panel informativo para el usuario sobre funcionalidad de transcripción de radio
st.info("""
### 📋 Cómo funciona esta aplicación:

**🎯 Objetivo**: Acceder y transcribir los mensajes de radio de las sesiones de F1.

**📌 Instrucciones**:
1. **Selecciona un Gran Premio** del desplegable.
2. **Elige una sesión** de ese Gran Premio para mostrar una tabla con los mensajes disponibles.
3. **Selecciona un mensaje de radio** por la hora en la que fue emitido.
4. **Podrás escuchar el audio** y pulsar en **Transcribir Audio** para obtener la transcripción.
""")

st.markdown("---")

# Configuración del dispositivo de cómputo (GPU si está disponible, CPU como fallback)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# CARGA DEL MODELO WHISPER: Caché del modelo de transcripción automática para eficiencia
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Carga el modelo Whisper medium de OpenAI para transcripción de audio a texto.
    Configurado con chunks de 30s y stride de 5s para mejor precisión en audio largo.
    """
    return pipeline("automatic-speech-recognition", 
                    "openai/whisper-medium", 
                    chunk_length_s=30, 
                    stride_length_s=5, 
                    return_timestamps=False, 
                    device=device)

# Inicializar modelo una sola vez por sesión
pipe = load_model()

# LAYOUT PRINCIPAL: División en dos columnas para selección y visualización
col1, col2 = st.columns(2)

with col1:
    # SELECCIÓN TEMPORAL: Año disponible para datos F1
    available_years = [2023, 2024, 2025]
    selected_year = st.selectbox("Selecciona un año:", available_years, index=len(available_years) - 1)

    # OBTENCIÓN DE DATOS: Cargar todos los Grandes Premios del año seleccionado
    # OBTENCIÓN DE DATOS: Cargar todos los Grandes Premios del año seleccionado
    all_meetings = fetch_data("meetings", {"year": selected_year})

    # Validación de disponibilidad de datos
    if all_meetings.empty:
        st.error("Datos no encontrados para este año.")
        st.stop()

    # FILTRADO GEOGRÁFICO: Selección por país para encontrar Gran Premio específico
    available_countries = sorted(all_meetings["country_name"].dropna().unique())
    selected_country = st.selectbox("Selecciona país:", available_countries)
    filtered_meetings = all_meetings[all_meetings["country_name"] == selected_country].copy()
    # Formateo de etiquetas legibles para el usuario
    filtered_meetings["label"] = filtered_meetings["meeting_name"] + " - " + filtered_meetings["location"]
    filtered_meetings = filtered_meetings.sort_values(by="meeting_key", ascending=False)

with col2:
    # SELECCIÓN DE EVENTO: Gran Premio específico y sus sesiones
    selected_meeting = st.selectbox("Selecciona Gran Premio:", filtered_meetings["label"], disabled=True)
    selected_meeting_key = filtered_meetings.loc[filtered_meetings["label"] == selected_meeting, "meeting_key"].values[0]
    
    # CARGA DE SESIONES: Obtener todas las sesiones disponibles (P1, P2, P3, Q, R)
    sessions = fetch_sessions(selected_meeting_key)
    selected_session = st.selectbox("Selecciona sesión:", sessions["label"])
    # Extracción del tipo de sesión para procesamiento
    sessions["session_type"] = sessions["label"].str.extract(r"^(.*?)\s\(")
    selected_session_type = sessions.loc[sessions["label"] == selected_session, "session_type"].values[0]
    selected_session_key = sessions.loc[sessions["label"] == selected_session, "session_key"].values[0]

# CARGA Y VISUALIZACIÓN DE MENSAJES DE RADIO
radio_data = fetch_radios(selected_session_key)
if radio_data.empty:
    st.warning("No se ha seleccionado sesión.")
else:
    # FORMATEO DE DATOS: Limpiar timestamps y columnas innecesarias
    radio_data["date"] = radio_data["date"].str.replace("T", " ").str.slice(0, 19)
    radio_data = radio_data.drop(columns=["meeting_key", "session_key"])
    st.write(radio_data)
    
    # SELECCIÓN DE MENSAJE: Usuario elige mensaje específico por timestamp
    selected_radio = st.selectbox("Selecciona Mensaje de Radio", radio_data["date"], index=0)
    radio_url = radio_data[radio_data["date"] == selected_radio]["recording_url"].values[0]

    # REPRODUCCIÓN DE AUDIO: Player integrado de Streamlit
    st.audio(radio_url, format="audio/mp3", start_time=0)

    # TRANSCRIPCIÓN AUTOMÁTICA: Procesamiento con Whisper
    if st.button("Transcribir Audio"):
        with st.spinner("Transcribiendo audio..."):
            # 1. Descargar y procesar archivo de audio desde URL
            audio_data = download_and_process_audio(radio_url)
            if audio_data is not None:
                try:
                    # 2. Aplicar modelo Whisper para transcripción en inglés
                    transcription = pipe(audio_data, generate_kwargs={"language": 'English', "task": 'transcribe'})
                    formatted_transcription = transcription['text'].strip()
                    
                    # 3. Mostrar resultado y opción de descarga
                    st.text_area(f"Transcripción:", value=formatted_transcription, height=150)
                    st.download_button("Descargar Transcripción", formatted_transcription, file_name="transcription.txt")
                except Exception as e:
                    st.error(f"Error en la transcripción: {str(e)}")
            else:
                st.error("Error al descargar el audio")
