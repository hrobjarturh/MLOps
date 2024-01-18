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


class GoogLeNet(nn.Module):

    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.model = models.googlenet(weights=None, init_weights=True, num_classes=10)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    my_app()
