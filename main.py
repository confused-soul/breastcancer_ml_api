from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from joblib import load
import json

X = [
    [17.99, 122.8, 1001, 0.2776, 0.3001, 0.1471],
    [20.57, 132.9, 1326, 0.07864, 0.0869, 0.07017],
    [19.69, 130, 1203, 0.1599, 0.1974, 0.1279],
    [11.42, 77.58, 386.1, 0.2839, 0.2414, 0.1052],
    [20.29, 135.1, 1297, 0.1328, 0.198, 0.1043],
    [12.45, 82.57, 477.1, 0.17, 0.1578, 0.08089],
    [18.25, 119.6, 1040, 0.109, 0.1127, 0.074],
    [13.71, 90.2, 577.9, 0.1645, 0.09366, 0.05985],
    [13, 87.5, 519.8, 0.1932, 0.1859, 0.09353],
    [12.46, 83.97, 475.9, 0.2396, 0.2273, 0.08543],
    [16.02, 102.7, 797.8, 0.06669, 0.03299, 0.03323],
    [15.78, 103.6, 781, 0.1292, 0.09954, 0.06606],
    [19.17, 132.4, 1123, 0.2458, 0.2065, 0.1118],
    [15.85, 103.7, 782.7, 0.1002, 0.09938, 0.05364],
    [13.73, 93.6, 578.3, 0.2293, 0.2128, 0.08025],
    [14.54, 96.73, 658.8, 0.1595, 0.1639, 0.07364],
    [14.68, 94.74, 684.5, 0.072, 0.07395, 0.05259],
    [16.13, 108.1, 798.8, 0.2022, 0.1722, 0.1028],
    [19.81, 130, 1260, 0.1027, 0.1479, 0.09498],
    [13.54, 87.46, 566.3, 0.08129, 0.06664, 0.04781],
    [13.08, 85.63, 520, 0.127, 0.04568, 0.0311],
    [9.504, 60.34, 273.9, 0.06492, 0.02956, 0.02076],
    [15.34, 102.5, 704.4, 0.2135, 0.2077, 0.09756],
    [21.16, 137.2, 1404, 0.1022, 0.1097, 0.08632],
    [16.65, 110, 904.6, 0.1457, 0.1525, 0.0917],
    [17.14, 116, 912.7, 0.2276, 0.2229, 0.1401],
    [14.58, 97.41, 644.8, 0.1868, 0.1425, 0.08783],
    [18.61, 122.1, 1094, 0.1066, 0.149, 0.07731],
    [15.3, 102.4, 732.4, 0.1697, 0.1683, 0.08751],
    [17.57, 115, 955.1, 0.1157, 0.09875, 0.07953],
    [18.63, 124.8, 1088, 0.1885, 0.1868, 0.0852],
    [11.84, 77.93, 440.6, 0.1109, 0.1516, 0.09333],
    [17.02, 113, 899.3, 0.09263, 0.06262, 0.08216],
]


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
    
    rad : float
    per : float
    area : float
    comp : float
    conc : float
    conp : float
    

# loading the saved model
diabetes_model = load('breastcancer_model.joblib')


@app.post('/breastcancer_prediction')
def diabetes_pred(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    rm = input_dictionary['red']
    pm = input_dictionary['per']
    am = input_dictionary['area']
    cm = input_dictionary['comp']
    cnm = input_dictionary['conc']
    cpm = input_dictionary['comp']


    input_list = scaler.transform([[rm, pm, am, cm, anm, cpm]])
    
    prediction = diabetes_model.predict(input_list)
    
    if prediction[0] == 0:
        return 'Benign'
    
    else:
        return 'Melignant'


