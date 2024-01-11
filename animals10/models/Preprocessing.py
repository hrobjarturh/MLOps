from torchvision import transforms

class Preprocessing:

    def preprocess_images(img, img_size):

        # Define the transformation
        transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(0, 1)
            ]
        )

        # Resize image
        img_resized = transform(img).unsqueeze(0)

        return img_resized