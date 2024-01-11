import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


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
    print("GoogLeNet")
    # def __init__(self):
    #     self.model = models.googlenet(pretrained=False, num_classes=10)
