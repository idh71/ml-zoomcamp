import numpy as np

import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray

from pydantic import BaseModel

class CreditApplication(BaseModel):
    name: str
    age: int
    country: str
    rating: float

# model_ref = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5")
model_ref = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5")
# dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("credit_risk_classifier", runners=[model_runner])


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
# async def classify(application_data):
def classify(application_data):
    #vector = dv.transform(application_data)
    # prediction = await model_runner.predict.async_run(vector)
    # prediction = model_runner.predict.async_run(application_data)
    prediction = model_runner.predict.run(application_data)
    # print(prediction)
    return prediction
    # result = prediction[0]

    # if result > 0.5:
    #     return {
    #         "status": "DECLINED"
    #     }
    # elif result > 0.25:
    #     return {
    #         "status": "MAYBE"
    #     }
    # else:
    #     return {
    #         "status": "APPROVED"
    #     }
