from torchvision import models


class GoogLeNet():
    def __init__(self):
        self.model = models.googlenet(pretrained=False, num_classes=10)

