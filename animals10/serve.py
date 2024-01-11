from typing import Annotated

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from animals10.models import GoogLeNet

MODELPATH = "models/googlenet_model.pth"

# create a pydantic base class that has an UploadFile object and an int called top_amount
class ImageRequest(BaseModel):
    file: UploadFile = File(...)
    top_amount: int = 1

# create a FastAPI app

app = FastAPI()

@app.get("/")
def read_root():
    return "Live"

@app.get("/ping")
def read_root():
    return "Pong"

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}


def load_image(image_path):
    image = Image.open(image_path)
    return image

# Create a function that preprocesses the image)
def preprocess_data(input_image):
    # TODO: Use preproccess module
    preprocess = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    return input_batch

# Create a function that loads the model and makes a prediction
def load_model():
    model = GoogLeNet().model
    model.load_state_dict(torch.load(MODELPATH))  
    return model

# Create a FastAPI endpoint that takes an image path as input and returns a prediction
@app.post("/predict")
async def predict_file(imageRequest:ImageRequest):
    print(f'predicting .. {imageRequest.file.filename}')
    model = load_model()
    image = load_image(imageRequest.file.filename)
    input_batch = preprocess_data(image)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    probabilities, labels = torch.topk(probabilities, imageRequest.top_amount)

    return {"Ouput": int(labels)}




