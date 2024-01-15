from torchvision import transforms


class Preprocessing:
    def preprocess_images(img, img_size):
        """
        Preprocesses input image for model prediction.

        Args:
            img (PIL.Image.Image): Input image to be preprocessed.
            img_size (int): Target size for resizing the image.

        Returns:
            torch.Tensor: Preprocessed image tensor suitable for model input.
        """
        # Define the transformation
        transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor(), transforms.Normalize(0, 1)]
        )

        # Resize image
        img_resized = transform(img).unsqueeze(0)

        return img_resized
