from fastapi import FastAPI
from pydantic import BaseModel

from animals10 import Predictor
from animals10.data.Preprocessing import Preprocessing

app = FastAPI()

predictor = Predictor("models/googlenet_model_5.pth")


class Data(BaseModel):
    image_path: str | list[str]

@app.post("/predict")
async def predict(data : Data):
    if isinstance(data.image_path, str):
        data.image_path = [data.image_path]

    results = predictor.predict(data.image_path)

    return {"result": results}

@app.get("/ping")
def ping():
    return "Pong"


