import os  # noqa: I001

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

from Preprocessing import Preprocessing

# Italian to English dictionary

it2en = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel",
}

IMAGE_SIZE = 224
RANDOM_SEED = 42
DATASET_BATCH_SIZE = 200

def collect_file_info(root_folder):
    # Get a list of all image files and categories
    image_names = []
    category_names = []

    # Loop through each subfolder in the root folder
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        # Skip if it's not a directory
        if not os.path.isdir(subfolder_path):
            print('no dir', subfolder_path)
            continue

        # Loop through each file in the subfolder
        for file_name in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file_name)

            # Append file and subfolder information to the lists
            image_names.append(file_name)
            category_names.append(subfolder)

    return image_names, category_names

def create_splits(image_names, category_names):

    # Split the data into training (70%), validation (15%), and test (15%) sets
    input_train, input_temp, target_train, target_temp = train_test_split(
        image_names, category_names, test_size=0.3, random_state=RANDOM_SEED
    )

    input_val, input_test, target_val, target_test = train_test_split(
        input_temp, target_temp, test_size=0.5, random_state=RANDOM_SEED
    )

    # Print the lengths of the sets
    print(f"Training set size: {len(input_train)}")
    print(f"Validation set size: {len(input_val)}")
    print(f"Test set size: {len(input_test)}")

    return input_train, target_train, input_val, target_val, input_test, target_test

def save_batches(set_type, inputs, targets):

    batch_counter = 0
    for input_batch,target_batch in zip(inputs,targets):
        data, labels = [], []
        batch_counter += 1
        image_counter = 0
        for image_idx in range(len(input_batch)):
            image = input_batch[image_idx]
            label = target_batch[image_idx]

            input_image_path = os.path.join(f"data/raw/archive/raw-img/{label}", image)

            img = Image.open(input_image_path).convert("RGB")

            # resize image and add to list of images
            img_resized = Preprocessing.preprocess_images(img, IMAGE_SIZE)
            data.append(img_resized)

            # get target label and add to list of labels
            target = list(it2en.keys()).index(label)
            labels.append(target)
            
            image_counter += 1
        
        print(f'{set_type}_{batch_counter}.pth')

        data = torch.cat(data)
        labels = torch.tensor(labels)

        print(data.shape)
        print(labels.shape)

        torch.save(torch.utils.data.TensorDataset(data, labels), F"data/processed/{set_type}/{set_type}_{batch_counter}.pt")

def check_folders():
    print('Checking foler structure ...')

    # Check if the "data/processed/" folder exists
    if not os.path.exists("data/processed/"):
        os.makedirs("data/processed/")

    # Check if the "test", "train" and "val" folders exist
    if not os.path.exists("data/processed/test"):
        os.makedirs("data/processed/test")
    if not os.path.exists("data/processed/train"):
        os.makedirs("data/processed/train")
    if not os.path.exists("data/processed/val"):
        os.makedirs("data/processed/val")

    print('Folder structure checked!')

if __name__ == "__main__":

    check_folders()

    # get all images and their class
    image_names, category_names = collect_file_info("data/raw/archive/raw-img/")

    # create sets
    input_train, target_train, input_val, target_val, input_test, target_test = create_splits(image_names, category_names)

    # Create sublists
    input_train = [input_train[i:i + DATASET_BATCH_SIZE] for i in range(0, len(input_train), DATASET_BATCH_SIZE)]
    target_train = [target_train[i:i + DATASET_BATCH_SIZE] for i in range(0, len(target_train), DATASET_BATCH_SIZE)]

    input_val = [input_val[i:i + DATASET_BATCH_SIZE] for i in range(0, len(input_val), DATASET_BATCH_SIZE)]
    target_val = [target_val[i:i + DATASET_BATCH_SIZE] for i in range(0, len(target_val), DATASET_BATCH_SIZE)]

    input_test = [input_test[i:i + DATASET_BATCH_SIZE] for i in range(0, len(input_test), DATASET_BATCH_SIZE)]
    target_test = [target_test[i:i + DATASET_BATCH_SIZE] for i in range(0, len(target_test), DATASET_BATCH_SIZE)]

    save_batches('train',input_train,target_train)
    save_batches('val',input_val,target_val)
    save_batches('test',input_test,target_test)
