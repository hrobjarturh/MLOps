import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf

from animals10.Loader import Loader
from animals10.models.GoogLeNet import GoogLeNet

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

        print(f"loading {self.hyperparams.training_batch} train batches")
        self.train_loader = Loader().load(
            hyperparams, batch_amount=hyperparams.training_batch, gcs_path = "gs://data-mlops-animals-10/data-mlops-animals10/data/processed/train/"
        )

        print(f"loading {self.hyperparams.validation_batch} val batches")
        self.val_loader = Loader().load(
            hyperparams, batch_amount=hyperparams.validation_batch, gcs_path = "gs://data-mlops-animals-10/data-mlops-animals10/data/processed/val/"
        )

    def train(self, log_to_wandb=True):
        """
        Trains the neural network model.

        Uses the specified loss function, optimizer, and hyperparameters for training.
        Logs training progress to WandB.
        """

        if log_to_wandb:
            wandb.init(project="MLOps", entity="naelr")
        training_loss = []
        validation_accuracies = []
        print("\nStarting training process ...")
        for epoch in range(self.hyperparams.epochs):
            self.model.train()
            epoch_loss = 0
            step_count = 0
            for inputs, labels in self.train_loader:
                start_time = time.time()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.logits, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                step_count += 1
                end_time = time.time()
                elapsed_time = end_time - start_time
                steps_left = len(self.train_loader) - step_count
                print(
                    f"Epoch={epoch}/{self.hyperparams.epochs}, step={step_count}/{len(self.train_loader)}, step time = {elapsed_time:.1f} secs, approximate time to finish = {(steps_left*elapsed_time)/60:.1f} mins",
                    end="\r",
                )
            avg_training_loss = epoch_loss / len(self.train_loader)
            training_loss.append(avg_training_loss)
            print(f"\nValidating ...")
            validation_accuracy = self.validate()
            validation_accuracies.append(validation_accuracy)

            # Logging to wandb
            if log_to_wandb:
                wandb.log({"Epoch": epoch, "Training loss": avg_training_loss})
                wandb.log({"Epoch": epoch, "Validation accuracy": validation_accuracy})
            print(
                f"Epoch [{epoch+1}/{self.hyperparams.epochs}], Loss: {avg_training_loss:.2f}, Validation accuracy: {validation_accuracy:.2f}"
            )
        return training_loss, validation_accuracies

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
            step_count = 0
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                step_count += 1
                print(f"step {step_count}/{len(self.val_loader)}", end="\r")

        accuracy = total_correct / total_samples
        return accuracy

    def save_model(self, filepath="gs://data-mlops-animals-10/data-mlops-animals10/data/models/"):
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


    path = "gs://data-mlops-animals-10/data-mlops-animals10/data/models/"
    if "googlenet_model_0.pth" not in path:

        newest_versions = 0
    else:
        versions = [int(file.split("_")[2].split(".")[0]) for file in path if file.startswith("googlenet_model_")]
        newest_versions = max(versions) + 1

    return f"gs://data-mlops-animals-10/data-mlops-animals10/data/models/googlenet_model_{newest_versions}.pth"


if __name__ == "__main__":
    print("Opening config files ...")
    with open("config/config.yaml", "r") as f:
        cfg = OmegaConf.load(f)
        hyperparams = instantiate(cfg.hyperparameters_training)

    print(f"System args: {sys.argv[1:]}")

    for hyperparam in sys.argv[1:]:
        hp, val = hyperparam.split("=")
        if hp in hyperparams.keys():
            try:
                val = eval(val)
            except NameError:
                pass
            hyperparams[hp] = val

    print("Initializing training with hyperparameters:")
    print(hyperparams)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoogLeNet().model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams.learning_rate)
    trainer = Trainer(model, device, criterion, optimizer, hyperparams)
    trainer.train()
    trainer.save_model(decide_filename())
