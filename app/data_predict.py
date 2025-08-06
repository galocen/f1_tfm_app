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

def predict_laptime(data, session):
    modelo, scaler = load_model(session)
    data_scaled = scaler.transform(data)
    predicciones = modelo.predict(data_scaled)
    return predicciones
