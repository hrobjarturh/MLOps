import os
import shutil
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf

from animals10.Loader import Loader
from animals10.models.GoogLeNet import GoogLeNet
from animals10.train_model import Trainer, decide_filename


@pytest.fixture(scope="session")
def hydra_config():
    config = OmegaConf.load("./config/config.yaml")
    # Solely for running the test faster
    config.hyperparameters_training.epochs = 1
    config.hyperparameters_training.training_batch = 1
    config.hyperparameters_training.validation_batch = 1
    return config


def test_data_loading(hydra_config):
    """
    Tests if data is loaded.
    """
    train_loader = Loader().load(
        hydra_config.hyperparameters_training,
        batch_amount=hydra_config.hyperparameters_training.training_batch,
        gcs_path = "gs://data-mlops-animals-10/data-mlops-animals10/data/processed/train",
    )
    validation_loader = Loader().load(
        hydra_config.hyperparameters_training,
        batch_amount=hydra_config.hyperparameters_training.validation_batch,
        gcs_path = "gs://data-mlops-animals-10/data-mlops-animals10/data/processed/val",
    )

    assert train_loader is not None, "Training data loader is None"
    assert validation_loader is not None, "Validation data loader is None"


def test_training_process(hydra_config):
    """
    Tests if the training process runs.
    """
    hyperparams = hydra_config.hyperparameters_training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoogLeNet().model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams.learning_rate)

    trainer = Trainer(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        hyperparams=hyperparams,
    )
    train_loss, val_acc = trainer.train(log_to_wandb=False)

    assert len(train_loss) == 1, "Training loss is not populated"
    assert len(val_acc) == 1, "Validation accuracies are not populated"


def setup_model_directory():
    """
    Creates a temporary directory for testing.
    """
    return tempfile.mkdtemp()


def create_model_files(directory, filenames):
    for filename in filenames:
        with open(os.path.join(directory, filename), "w") as f:
            f.write("")  # Create an empty file


def test_initial_filename_creation():
    """
    Tests if the initial filename is created correctly.
    """
    dirpath = setup_model_directory()
    try:
        # Test with an empty temporary directory
        expected_filename = f"{dirpath}/googlenet_model_0.pth"
        result = decide_filename(dirpath)
        assert result == expected_filename, f"Test failed: {result} != {expected_filename}"

    finally:
        # Clean up by removing the temporary directory
        shutil.rmtree(dirpath)


def test_incrementing_filename_version():
    """
    Tests if the version number is incremented correctly.
    """
    dirpath = setup_model_directory()
    try:
        create_model_files(dirpath, ["googlenet_model_0.pth", "googlenet_model_1.pth", "googlenet_model_2.pth"])
        assert decide_filename(dirpath) == f"{dirpath}/googlenet_model_3.pth"
    finally:
        # Clean up by removing the temporary directory
        shutil.rmtree(dirpath)
