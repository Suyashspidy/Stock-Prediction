from fastapi import FastAPI
import numpy as np
import joblib
import pandas as pd

model = joblib.load('app/rf_model.joblib')

class_names = np.array(['Crude_oil_Price'])

app = FastAPI()

@app.get('/')
def read_root():
    return {'message': 'Stock Prediction model API'}

@app.post('/predict')
def predict(data: dict):
         
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    class_name = class_names[prediction][0]
    return {'predicted_class': class_name}