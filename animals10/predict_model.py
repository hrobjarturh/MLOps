from typing import List, Union
import torch
from PIL import Image
from animals10.data.Preprocessing import Preprocessing
from animals10.models.GoogLeNet import GoogLeNet


class Predictor:
    def __init__(self, model_path="models/googlenet_model_0.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)

    def load_model(self, model_path: str):
        model = GoogLeNet().model
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def predict(self, image_paths: Union[str, List[str]], top_amount=1):
        IMAGE_SIZE = 224
        if isinstance(image_paths, str):
            # If a single path is provided, convert it to a list
            image_paths = [image_paths]

        results = []

        for image_path in image_paths:
            input_image = Image.open(image_path).convert("RGB")
            input_batch = Preprocessing.preprocess_images(input_image, IMAGE_SIZE)
            input_batch = input_batch.to(self.device)

            with torch.no_grad():
                output = self.model(input_batch)

            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            probabilities, labels = torch.topk(probabilities, top_amount)

            results.append((float(probabilities.cpu()), int(labels.cpu())))

        return results