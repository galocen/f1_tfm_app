import pickle

def load_model(session):
    if session == 'Q1':
        with open('./models/model_Q1.pkl', 'rb') as f:
            modelo_cargado = pickle.load(f)
        with open('./models/scaler_Q1.pkl', 'rb') as f:
            scaler_cargado = pickle.load(f)

    elif session == 'Q2':
        with open('./models/model_Q2.pkl', 'rb') as f:
            modelo_cargado = pickle.load(f)
        with open('./models/scaler_Q2.pkl', 'rb') as f:
            scaler_cargado = pickle.load(f)

    elif session == 'Q3':
        with open('./models/model_Q3.pkl', 'rb') as f:
            modelo_cargado = pickle.load(f)
        with open('./models/scaler_Q3.pkl', 'rb') as f:
            scaler_cargado = pickle.load(f)
    
    return modelo_cargado, scaler_cargado

def get_expected_features(session):
    """Obtiene los nombres de las características esperadas por el modelo"""
    try:
        _, scaler = load_model(session)
        if hasattr(scaler, 'feature_names_in_'):
            return scaler.feature_names_in_.tolist()
        else:
            return None
    except:
        return None

def predict_laptime(data, session):
    modelo, scaler = load_model(session)
    
    # Verificar si el scaler tiene nombres de características
    if hasattr(scaler, 'feature_names_in_'):
        expected_features = scaler.feature_names_in_.tolist()
        
        # Reordenar las columnas según el orden esperado
        data = data[expected_features]
    
    data_scaled = scaler.transform(data)
    predicciones = modelo.predict(data_scaled)
    return predicciones
