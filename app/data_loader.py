import streamlit as st
import requests
import pandas as pd
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

@st.cache_data
def fetch_sessions(meeting_key):
    # Devuelve las sesiones de un gran premio específico.
    df = fetch_data("sessions", {"meeting_key": meeting_key})

    # Combina el nombre de la sesión y la fecha de inicio para mostrarlo en pantalla
    df["label"] = df["session_name"] + " (" + df["date_start"] + ")"

    return df[["session_key", "label"]].drop_duplicates()

@st.cache_data
def fetch_radios(session_key):
    # Devuelve los mensajes de radio para una sesión específica.
    return fetch_data("team_radio", {"session_key": session_key})
