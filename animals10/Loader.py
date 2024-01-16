import torch
from torch.utils.data import ConcatDataset, DataLoader


# Loader class
class Loader:
    def __init__(self):
        self.data_loader = None

    def load(self, hyperparams, batch_amount, folder_path):
        # List to store individual datasets
        datasets = []
        batch_counter = 0
        # Loop through the files in the folder
        for filename in folder_path:
            batch_counter += 1

            file_path = folder_path + filename

            # Check if the file is a torch DataLoader object
            if filename.endswith(".pt"):
                # Load the DataLoader object from the file
                dataset = torch.load(file_path)

                # Append the dataset to the list
                datasets.append(dataset)

            if batch_counter >= batch_amount and "train/" in folder_path:
                break

        # Check if any DataLoader objects were loaded
        if not datasets:
            print("No DataLoader objects found in the folder.")
        else:
            # Concatenate the datasets into a single dataset
            concatenated_dataset = ConcatDataset(datasets)

            # Create a DataLoader for the concatenated dataset
            concatenated_dataloader = DataLoader(concatenated_dataset, hyperparams.batch_size, shuffle=True)
            print(f"Finished loading {folder_path}")
            return concatenated_dataloader
        return None
