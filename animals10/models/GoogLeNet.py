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


class GoogLeNet:
    def __init__(self):
        self.model = models.googlenet(weights=None, init_weights=True, num_classes=10)


if __name__ == "__main__":
    my_app()


class GoogLeNet:
    def __init__(self):
        self.model = models.googlenet(weights=None, init_weights=True, num_classes=10)
