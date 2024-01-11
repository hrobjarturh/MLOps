import os

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from models.GoogLeNet import GoogLeNet

# TODO: Add logger


class Trainer:
    def __init__(self, model, device, criterion, optimizer, hyperparams) -> None:
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.hyperparams = hyperparams

    def load(self, filepath="/data/processed/dataset.pt", local=True):
        if local:
            try:
                dataset = torch.load(filepath)
            except FileNotFoundError:
                print(f"Error: File '{filepath}' not found.")
                exit()

            self.data_loader = DataLoader(dataset, self.hyperparams.batch_size, shuffle=True)
        else:
            # TODO: Load data with dvs
            self.data_loader = None

    def train(self):
        for epoch in range(self.hyperparams.epochs):
            self.model.train()

            for inputs, labels in self.data_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.logits, labels)
                loss.backward()
                self.optimizer.step()

            print(f"Epoch [{epoch+1}/{self.hyperparams.epochs}], Loss: {loss.item():.4f}")  # TODO: logger

    def validate(self, val_loader):  # TODO: Create validation set
        self.model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    def save_model(self, filepath="models/googlenet_model.pth"):
        torch.save(self.model.state_dict(), filepath)

def decide_filename():
    files = os.listdir('models')
    if "googlenet_model_0.pth" not in files:
        newest_versions = 0
    else:
        versions = [int(file.split("_")[2].split(".")[0]) for file in files if file.startswith('googlenet_model_')]
        newest_versions = max(versions) + 1

    return f"models/googlenet_model_{newest_versions}.pth"

if __name__ == "__main__":

    print('Opening config files ...')
    with open("config/config.yaml", "r") as f:
        cfg = OmegaConf.load(f)
        hyperparams = instantiate(cfg.hyperparams)


    print('Training ...')
    model = GoogLeNet().model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=hyperparams.learning_rate)

    trainer = Trainer(model, device, criterion, optimizer, hyperparams)


    trainer.load(filepath="data/processed/dataset.pt",)
    trainer.train()
    
    files = os.listdir('models')
    if "googlenet_model_0.pth" not in files:
        newest_versions = 0
    else:
        versions = [int(file.split("_")[2].split(".")[0]) for file in files if file.startswith('googlenet_model_')]
        newest_versions = max(versions) + 1

    trainer.save_model(decide_filename())
