from fastapi import FastAPI
from pydantic import BaseModel

from animals10.predict_model import Predictor

app = FastAPI()

# On macOS, the path to the model is different
predictor = Predictor("models/googlenet_model_0.pth")


class Data(BaseModel):
    image_path: str | list[str]


@app.post("/predict")
async def predict(data: Data):
    if isinstance(data.image_path, str):
        data.image_path = [data.image_path]

    results = predictor.predict(data.image_path)

    return {"result": results}


@app.get("/ping")
def ping():
    return "Pong"

@app.get("/")
def welcome():
    return "Status: Live"
