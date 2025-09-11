import streamlit as st
from transformers import pipeline
import torch
import numpy as np
from app.data_loader import (
    fetch_data,
    fetch_sessions,
    fetch_radios
)
from app.data_processor import download_and_process_audio

st.set_page_config(page_title="Mensajes de Radio", layout="wide", page_icon="📻")

st.title('Mensajes de Radio')
st.markdown("Seleccione una sesión de un gran premio para acceder a los mensajes de radio y posteriormente seleccione el deseado para poder escucharlo y mostrar su transcripción.")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_model():
    return pipeline("automatic-speech-recognition", 
                    "openai/whisper-medium", 
                    chunk_length_s=30, 
                    stride_length_s=5, 
                    return_timestamps=False, 
                    device=device)

pipe = load_model()

col1, col2 = st.columns(2)

with col1:
    available_years = [2023, 2024, 2025]
    selected_year = st.selectbox("Selecciona un año:", available_years, index=len(available_years) - 1)

    all_meetings = fetch_data("meetings", {"year": selected_year})

    if all_meetings.empty:
        st.error("Datos no encontrados para este año.")
        st.stop()

    available_countries = sorted(all_meetings["country_name"].dropna().unique())
    selected_country = st.selectbox("Selecciona país:", available_countries)
    filtered_meetings = all_meetings[all_meetings["country_name"] == selected_country].copy()
    filtered_meetings["label"] = filtered_meetings["meeting_name"] + " - " + filtered_meetings["location"]
    filtered_meetings = filtered_meetings.sort_values(by="meeting_key", ascending=False)

with col2:
    selected_meeting = st.selectbox("Selecciona Gran Premio:", filtered_meetings["label"], disabled=True)
    selected_meeting_key = filtered_meetings.loc[filtered_meetings["label"] == selected_meeting, "meeting_key"].values[0]
    sessions = fetch_sessions(selected_meeting_key)
    selected_session = st.selectbox("Selecciona sesión:", sessions["label"])
    sessions["session_type"] = sessions["label"].str.extract(r"^(.*?)\s\(")
    selected_session_type = sessions.loc[sessions["label"] == selected_session, "session_type"].values[0]
    selected_session_key = sessions.loc[sessions["label"] == selected_session, "session_key"].values[0]

radio_data = fetch_radios(selected_session_key)
if radio_data.empty:
    st.warning("No se ha seleccionado sesión.")
else:
    radio_data["date"] = radio_data["date"].str.replace("T", " ").str.slice(0, 19)
    radio_data = radio_data.drop(columns=["meeting_key", "session_key"])
    st.write(radio_data)
    selected_radio = st.selectbox("Selecciona Mensaje de Radio", radio_data["date"], index=0)
    radio_url = radio_data[radio_data["date"] == selected_radio]["recording_url"].values[0]

    st.audio(radio_url, format="audio/mp3", start_time=0)

    if st.button("Transcribir Audio"):
        with st.spinner("Transcribiendo audio..."):
            audio_data = download_and_process_audio(radio_url)
            if audio_data is not None:
                try:
                    transcription = pipe(audio_data, generate_kwargs={"language": 'English', "task": 'transcribe'})
                    formatted_transcription = transcription['text'].strip()
                    
                    st.text_area(f"Transcripción:", value=formatted_transcription, height=150)
                    st.download_button("Descargar Transcripción", formatted_transcription, file_name="transcription.txt")
                except Exception as e:
                    st.error(f"Error en la transcripción: {str(e)}")
            else:
                st.error("Error al descargar el audio")
