import io

import torch
from google.cloud import storage
from torch.utils.data import ConcatDataset, DataLoader


# Loader class
class Loader:
    def __init__(self):
        self.data_loader = None

    def load(self, hyperparams, batch_amount, gcs_path):
        client = storage.Client()
        bucket_name, prefix = self.parse_gcs_path(gcs_path)
        bucket = client.bucket(bucket_name)

        print("Client", {client})
        print("Bucket Name", {bucket_name})
        print("Prefix", {prefix})
        print("Bucket", {bucket})

        # List to store individual datasets
        datasets = []
        batch_counter = 0

        blobs = client.list_blobs(bucket, prefix=prefix)

        for blob in blobs:
            if blob.name.endswith(".pt"):
                print(f"Loading {blob.name} from bucket {bucket_name}")
                batch_counter += 1
                file_data = blob.download_as_bytes()

                # Load the dataset from bytes
                dataset = torch.load(io.BytesIO(file_data))
                datasets.append(dataset)

                if batch_counter >= batch_amount and (("train/" in blob.name) or ("val/" in blob.name)):
                    break

        if not datasets:
            print("No DataLoader objects found in the bucket.")
        else:
            concatenated_dataset = ConcatDataset(datasets)
            concatenated_dataloader = DataLoader(concatenated_dataset, hyperparams.batch_size, shuffle=True)
            print(f"Finished loading from {gcs_path}")
            return concatenated_dataloader

    def parse_gcs_path(self, gcs_path):
        path_parts = gcs_path.replace("gs://", "").split("/")
        bucket_name = path_parts[0]
        prefix = "/".join(path_parts[1:])
        return bucket_name, prefix
