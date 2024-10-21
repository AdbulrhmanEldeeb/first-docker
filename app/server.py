from fastapi import FastAPI , Query 
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Annotated 

class Inputs(BaseModel):
    value1: Annotated[float , Query(gt=0)]
    value2: Annotated[float , Query(gt=0)]
    value3: Annotated[float , Query(gt=0)]
    value4: Annotated[float , Query(gt=0)]
    class Config : 
        extra="forbid"

class_names = np.array(['setosa', 'versicolor', 'virginica'])

app = FastAPI()


@app.get("/")
async def welcome():
    return {"message": "welcome"}

loaded_rf = joblib.load("app/rf_model.joblib")


@app.post("/predict", response_model=dict)
async def classify(inputs: Inputs) -> dict:
    input_values = np.array(
        [[inputs.value1, inputs.value2, inputs.value3, inputs.value4]], dtype=np.float16
    )
    result = class_names[loaded_rf.predict(input_values)[0]]

    return {"prediction": result}
