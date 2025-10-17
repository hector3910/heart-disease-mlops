from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


model = joblib.load("app/model.joblib")

app = FastAPI()


class Input(BaseModel):
    features: list


@app.post("/predict")
def predict(data: Input):
    columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
               'FastingBS', 'RestingECG', 'MaxHR',
               'ExerciseAngina', 'Oldpeak', 'ST_Slope']
    X = pd.DataFrame([data.features], columns=columns)
    proba = model.predict_proba(X)[0][1]
    return {"heart_disease_probability": proba,
            "prediction": int(proba > 0.5)}
    