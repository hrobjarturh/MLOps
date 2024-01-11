import torch
from torchvision import transforms

from animals10.models.GoogLeNet import GoogLeNet


class Predictor:
    def __init__(self, model_path="models/googlenet_model.pth"):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = GoogLeNet().model
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def preprocess_data(self, input_image):
        preprocess = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
        return input_batch

    def predict(self, input_image, top_amount=1):
        input_batch = self.preprocess_data(input_image)

        with torch.no_grad():
            output = self.model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        probabilities, labels = torch.topk(probabilities, top_amount)

        return probabilities, labels
