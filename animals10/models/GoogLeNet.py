
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision import datasets, models


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

class GoogLeNet():
    def __init__(self):
        self.model = models.googlenet(weights=None, init_weights=True, num_classes=10)

# data_transforms = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Assuming your dataset is structured in a directory with subfolders for each class
# dataset = datasets.ImageFolder(root='', transform=data_transforms)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


if __name__ == "__main__":
    my_app()

class GoogLeNet():
    def __init__(self):
        self.model = models.googlenet(weights=None, init_weights=True, num_classes=10)
