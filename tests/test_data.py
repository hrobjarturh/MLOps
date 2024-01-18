from omegaconf import OmegaConf

from animals10.Loader import Loader


def load_data(dataset_type):
    """
    Loads the data and returns the data loaders, the number of training and validation samples,
    the shape of the data, and the number of labels.
    """
    print("Running dataset tests...")

    # Load the hyperparameters
    cfg = OmegaConf.load("./config/config.yaml")
    hyperparameters = cfg.hyperparameters_training

    data_shape = [3, 224, 224]
    num_labels = 10

    # Initialize the loaders
    if dataset_type == "train":
        train_loader = Loader().load(
            hyperparameters, batch_amount = 5, gcs_path = "gs://data-mlops-animals-10/data-mlops-animals10/data/processed/train")
        return train_loader, data_shape, num_labels

    elif dataset_type == "validation":
        validation_loader = Loader().load(
            hyperparameters, batch_amount = 5, gcs_path = "gs://data-mlops-animals-10/data-mlops-animals10/data/processed/val")
        return validation_loader, data_shape, num_labels


def validate_dataset(loader, data_shape, num_labels, dataset_type):
    """
    Validates the dataset by checking the number of samples, the shape of the data, and the number of labels.
    """

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
    train_loader, data_shape, num_labels = load_data("train")
    validate_dataset(train_loader, data_shape, num_labels, "Training")


def test_validation_data():
    """
    Tests the validation dataset.
    """
    validation_loader, data_shape, num_labels = load_data("validation")
    validate_dataset(validation_loader, data_shape, num_labels, "Validation")


test_training_data()
test_validation_data()
