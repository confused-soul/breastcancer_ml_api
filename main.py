from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from joblib import load
import json
import numpy as np
import pandas as pd

dataset = pd.read_csv('breastcancer.csv')
X = dataset.drop(columns = 'Outcome', axis = 1)
columns_to_drop = ['tm', 'sm', 'sym', 'fdm', 'rse', 'tse', 'pse', 'ase', 'sse', 'cse', 'cnse', 'cpse', 'symse', 'fdse', 'rw', 'tw', 'pw', 'aw', 'sw', 'cw', 'cnw', 'cpw', 'symw', 'fdw']
X = X.drop(columns=columns_to_drop)

scaler = StandardScaler()
scaler.fit(X)


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    
    radius : float
    perimeter : float
    area : float
    compactness : float
    concativity : float
    concave_points : float
    

# loading the saved model
diabetes_model = load('breastcancer_model.joblib')


@app.post('/breastcancer_prediction')
def diabetes_pred(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    rm = input_dictionary['radius']
    pm = input_dictionary['perimeter']
    am = input_dictionary['area']
    cm = input_dictionary['compactness']
    cnm = input_dictionary['concativity']
    cpm = input_dictionary['concave_points']


    input_list = scaler.transform([[rm, pm, am, cm, anm, cpm]])
    
    prediction = diabetes_model.predict(input_list)
    
    if prediction[0] == 0:
        return 'Benign'
    
    else:
        return 'Melignant'


