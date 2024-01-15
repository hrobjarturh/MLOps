import os

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader

from Loader import Loader
from models.GoogLeNet import GoogLeNet

# TODO: Add logger


class Trainer:
    def __init__(self, model, device, criterion, optimizer, hyperparams) -> None:
        """
        Initializes the Trainer object.

        Args:
            model (torchvision.models.googlenet.GoogLeNet): The neural network model to be trained.
            device (torch.device): The device (CPU or GPU) on which the training will be performed.
            criterion: The loss function used for training.
            optimizer: The optimization algorithm used for updating the model's weights.
            hyperparams: An object containing hyperparameters for training.
        """

        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.hyperparams = hyperparams
        self.train_loader = Loader().load(
            hyperparams, batch_amount=hyperparams.training_batch, folder_path="data/processed/train"
        )
        self.val_loader = Loader().load(
            hyperparams, batch_amount=hyperparams.validation_batch, folder_path="data/processed/val"
        )

    def train(self):
        """
        Trains the neural network model.

        Uses the specified loss function, optimizer, and hyperparameters for training.
        Logs training progress to WandB.
        """

        wandb.init(project="MLOps", entity="naelr")
        training_loss = []
        validation_accuracies = []
        print("\n Starting training process ...")
        for epoch in range(self.hyperparams.epochs):
            self.model.train()
            epoch_loss = 0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.logits, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_training_loss = epoch_loss / len(self.train_loader)
            training_loss.append(avg_training_loss)
            validation_accuracy = self.validate()
            validation_accuracies.append(validation_accuracy)

            # Logging to wandb
            wandb.log({"Epoch": epoch, "Training loss": avg_training_loss})
            wandb.log({"Epoch": epoch, "Validation accuracy": validation_accuracy})
            print(
                f"Epoch [{epoch+1}/{self.hyperparams.epochs}], Loss: {avg_training_loss:.2f}, Validation accuracy: {validation_accuracy:.2f}"
            )

    def validate(self):
        """
        Validates the neural network model on the validation set.

        Returns:
            float: Validation accuracy.
        """

        self.model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        return accuracy

    def save_model(self, filepath="models/googlenet_model.pth"):
        """
        Saves the trained model's state dictionary to a file.

        Args:
            filepath (str): Path to save the model file.
        """

        torch.save(self.model.state_dict(), filepath)


def decide_filename():
    """
    Decides the filename for a new version of the GoogleNet model.

    Checks the existing models in the 'models' folder, extracts the version numbers
    from filenames, and generates a new filename with an incremented version number.

    Returns:
        str: The filename for the new version of the GoogleNet model.
    """

    files = os.listdir("models")
    if "googlenet_model_0.pth" not in files:
        newest_versions = 0
    else:
        versions = [int(file.split("_")[2].split(".")[0]) for file in files if file.startswith("googlenet_model_")]
        newest_versions = max(versions) + 1

    return f"models/googlenet_model_{newest_versions}.pth"


if __name__ == "__main__":
    print("Opening config files ...")
    with open("config/config.yaml", "r") as f:
        cfg = OmegaConf.load(f)
        hyperparams = instantiate(cfg.hyperparameters_training)

    print("Training ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoogLeNet().model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams.learning_rate)
    trainer = Trainer(model, device, criterion, optimizer, hyperparams)
    trainer.train()
    trainer.save_model(decide_filename())
