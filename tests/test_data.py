from omegaconf import OmegaConf

from animals10.Loader import Loader


def load_data():
    """
    Loads the data and returns the data loaders, the number of training and validation samples,
    the shape of the data, and the number of labels.
    """
    print("Running dataset tests...")

    # Load the hyperparameters
    cfg = OmegaConf.load("./config/config.yaml")
    hyperparameters = cfg.hyperparameters_training

    # Initialize the loaders
    train_loader = Loader().load(
        hyperparameters, batch_amount=hyperparameters.training_batch, folder_path="data/processed/train"
    )
    validation_loader = Loader().load(
        hyperparameters, batch_amount=hyperparameters.validation_batch, folder_path="data/processed/val"
    )

    N_train = 18325
    N_val = 3927
    data_shape = [3, 224, 224]
    num_labels = 10

    return train_loader, validation_loader, N_train, N_val, data_shape, num_labels


def validate_dataset(loader, N_samples, data_shape, num_labels, dataset_type="Training"):
    """
    Validates the dataset by checking the number of samples, the shape of the data, and the number of labels.
    """
    samples = len(loader.dataset)
    assert samples == N_samples, f"{dataset_type} dataset containing {samples} samples, expected {N_samples}"

    # Initialize a set to store unique labels
    unique_labels = set()

    # Iterate over batches in the data loader
    for batch, labels in loader:
        # Check the shape of each data sample in the batch
        for sample in batch:
            assert (
                list(sample.shape) == data_shape
            ), f"Data shape in {dataset_type} dataset is {sample.shape}, expected {data_shape}"

        # Add labels to the unique labels set
        unique_labels.update(labels.tolist())

    # Check if all labels are represented
    assert (
        len(unique_labels) == num_labels
    ), f"Number of unique labels in {dataset_type} dataset are {len(unique_labels)}, expected {num_labels}"

    print(f"{dataset_type} dataset test passed!")


def test_training_data():
    """
    Tests the training dataset.
    """
    train_loader, _, N_train, _, data_shape, num_labels = load_data()
    validate_dataset(train_loader, N_train, data_shape, num_labels, "Training")


def test_validation_data():
    """
    Tests the validation dataset.
    """
    _, validation_loader, _, N_val, data_shape, num_labels = load_data()
    validate_dataset(validation_loader, N_val, data_shape, num_labels, "Validation")
