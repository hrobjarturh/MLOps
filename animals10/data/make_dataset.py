import os

import torch
from PIL import Image
from torchvision import transforms
from animals10.models.GoogLeNet import GoogLeNet
from animals10.models.Preprocessing import Preprocessing

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


if __name__ == "__main__":
    # list of images as tensors and labels
    data, labels = [], []
    for animal in it2en.keys():
        input_folder = f"data/raw/archive/raw-img/{animal}/"

        print(f"processing '{it2en[animal]}'")
        count = 0
        image_tensors = []
        num_imgs = len(os.listdir(input_folder))

        # Loop through all files in the input folder
        for file_name in os.listdir(input_folder):
            # Check if the file is an image (you can customize the extensions as needed)
            if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                # Construct the full path of the input image
                input_image_path = os.path.join(input_folder, file_name)

                # open image and convert to grayscale
                img = Image.open(input_image_path).convert("RGB")

                IMAGE_SIZE = 224
                # Define the transformation
                transform = transforms.Compose(
                    [
                        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                        transforms.ToTensor(),
                        transforms.Normalize(0.5, 0.5)
                    ]
                )

                # resize image and add to list of images
                img_resized = Preprocessing.preprocess_images(img, IMAGE_SIZE)
                data.append(img_resized)

                # get target label and add to list of labels
                target = list(it2en.keys()).index(animal)
                labels.append(target)

                print(f"{count}/{num_imgs}", end="\r")
                count += 1

            if count >= 10:  # TODO: remove (only for dev purposes)
                break

    data = torch.cat(data)
    labels = torch.tensor(labels)

    print(data.shape)
    print(labels.shape)

    torch.save(torch.utils.data.TensorDataset(data, labels), "data/processed/dataset.pt")
