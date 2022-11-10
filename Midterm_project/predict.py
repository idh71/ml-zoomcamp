import numpy as np

import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray

import xgboost as xgb

from pydantic import BaseModel

class PersonalInfo(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
    us_native: str




   
model_ref = bentoml.xgboost.get("over_50k_model:latest")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("over_50k_classifier", runners=[model_runner])

@svc.api(input=JSON(), output=JSON())
#@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
# async def classify(application_data):
def classify(application_data):
    vector = dv.transform(application_data)
    
    prediction = model_runner.predict.run(vector)
    
    #return prediction
    

    if prediction >= 0.5:
        return {
            "status": "OVER 50K"
        }
  
    else:
        return {
            "status": "UNDER 50K"
        }