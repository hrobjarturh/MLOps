import io
from io import BytesIO
from typing import List

import torch
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image
from pydantic import BaseModel

from animals10.predict_model import Predictor

app = FastAPI()

# On macOS, the path to the model is different
predictor = Predictor("models/googlenet_model_0.pth")

@app.get("/ping")
def ping():
    return "Pong"

@app.get("/")
def welcome():
    return "Status: Live"

class TensorInput(BaseModel):
    data: List[List[List[List[float]]]]

@app.post("/process_tensor")
async def process_tensor(tensor_input: TensorInput):
    # Convert the received data to a torch tensor
    try:
        tensor_data = torch.tensor(tensor_input.data)
        # Check if the tensor has the expected shape
        if tensor_data.shape != (1, 3, 224, 224):
            raise ValueError("Invalid tensor shape. Expected torch.Size([1, 3, 224, 224])")
        
        results = predictor.predict(tensor_data)
        
        return {"results": results}


    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
