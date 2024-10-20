from fastapi import FastAPI 
from pydantic import BaseModel 
import joblib
import numpy as np
class Inputs(BaseModel): 
    value1:float  
    value2:float  
    value3:float  
    value4:float 

app=FastAPI() 

@app.get('/')
async def welcome(): 
    return {
        'message':"welcome"
            }
loaded_rf=joblib.load(r'app\rf_model.joblib')
@app.post('/predict',response_model=dict)
async def classify(inputs:Inputs)->dict:
    input_values=np.array([[inputs.value1,inputs.value2,inputs.value3,inputs.value4]],dtype=np.float16)
    result=loaded_rf.predict(input_values)[0]

    return {'prediction':int(result)}




