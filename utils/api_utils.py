import torch
from torchvision import transforms as transforms
import torch.nn.functional as functional
from PIL import Image
import os


def process_images_and_compute_similarity(model, file1, file2):
    def load_image_from_file(file, transformation, save_path=None):
        image = Image.open(file).convert('RGB')
        if transformation is not None:
            image = transformation(image)
            if save_path:
                # Convert tensor back to PIL Image to save file
                save_image = transforms.ToPILImage()(image)
                save_image.save(save_path)
            image = image.unsqueeze(0)
        return image

    transform = transforms.Compose([
        transforms.Resize((112, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_flipped = transforms.Compose([
        transforms.Resize((112, 96)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    os.makedirs("test", exist_ok=True)

    # Define paths for saving images
    image1_path = os.path.join("test", "image1_transformed.jpg")
    image2_path = os.path.join("test", "image2_transformed.jpg")
    image1_flipped_path = os.path.join("test", "image1_flipped.jpg")
    image2_flipped_path = os.path.join("test", "image2_flipped.jpg")

    # Load and transform images, then save them
    image1 = load_image_from_file(file1, transform, image1_path)
    image2 = load_image_from_file(file2, transform, image2_path)
    image1_flipped = load_image_from_file(file1, transform_flipped, image1_flipped_path)
    image2_flipped = load_image_from_file(file2, transform_flipped, image2_flipped_path)

    with torch.no_grad():
        features1 = model(image1, return_embeddings=True)
        features2 = model(image2, return_embeddings=True)
        features1_flipped = model(image1_flipped, return_embeddings=True)
        features2_flipped = model(image2_flipped, return_embeddings=True)

    similarity = functional.cosine_similarity(features1 + features1_flipped, features2 + features2_flipped)
    return similarity.item()
