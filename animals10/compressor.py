import os

import numpy as np
import torch


class FileCompressor:
    def __init__(self, directory):
        self.directory = directory

    def pt_to_npz(self):
        # Iterate over all files in the directory
        for filename in os.listdir(self.directory):
            if filename.endswith(".pt"):
                file_path = os.path.join(self.directory, filename)

                # Load the TensorDataset
                dataset = torch.load(file_path)

                # Check if it's a TensorDataset instance
                if isinstance(dataset, torch.utils.data.TensorDataset):
                    # Extract all tensors from the TensorDataset
                    tensors = [tensor.numpy() for tensor in dataset.tensors]
                else:
                    # If not a TensorDataset, skip to the next file
                    continue

                # Compress and save the NumPy arrays
                compressed_file_path = file_path.replace(".pt", ".npz")
                np.savez_compressed(compressed_file_path, *tensors)

                # Remove the original .pt file
                os.remove(file_path)

        print("Compression of .pt files complete.")

    def npz_to_pt(self):
        # Iterate over all files in the directory
        for filename in os.listdir(self.directory):
            if filename.endswith(".npz"):
                npz_file_path = os.path.join(self.directory, filename)

                # Load the compressed .npz file
                with np.load(npz_file_path) as data:
                    # Extract all arrays (assuming they were saved as array_0, array_1, etc.)
                    arrays = [data[f"arr_{i}"] for i in range(len(data))]

                    # Convert arrays to PyTorch tensors
                    tensors = [torch.from_numpy(arr) for arr in arrays]

                    # Save as a .pt file
                    pt_file_path = npz_file_path.replace(".npz", ".pt")
                    torch.save(torch.utils.data.TensorDataset(*tensors), pt_file_path)

                # Remove the original .npz file
                os.remove(npz_file_path)

        print("Conversion of .npz files complete.")


# Usage
converter = FileCompressor("./data/")
converter.pt_to_npz()  # To convert .pt to .npz and remove .pt
# converter.npz_to_pt()  # To convert .npz back to .pt and remove .npz
