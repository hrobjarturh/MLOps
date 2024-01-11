import hydra
from omegaconf import DictConfig, OmegaConf
from torchvision import models


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

class GoogLeNet():
    def __init__(self):
        self.model = models.googlenet(weights=None, init_weights=True, num_classes=10)

if __name__ == "__main__":
    my_app()
