import torch
from torchvision import transforms as transforms
import torch.nn.functional as functional
from PIL import Image
from dal import DAL
from meta import AGEDB, WANDB
import os


def process_images_and_compute_similarity(file1, file2):
    def load_image_from_file(file, transformation):
        image = Image.open(file).convert('RGB')
        if transformation is not None:
            image = transformation(image).unsqueeze(0)
        return image

    transform = transforms.Compose([
        transforms.Resize((112, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_flipped = transforms.Compose([
        transforms.Resize((112, 96)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    config = WANDB
    dataset = AGEDB
    model = DAL(
        loss_head=config["loss_head"],
        num_classes=dataset["num_classes"],
        embedding_dimension=config["embedding_dimension"],
    )

    model.load_state_dict(torch.load(config["model"]))
    print(f'Loaded weights from {config["model"]}')
    model.eval()

    image1 = load_image_from_file(file1, transform)
    image2 = load_image_from_file(file2, transform)

    image1_flipped = load_image_from_file(file1, transform_flipped)
    image2_flipped = load_image_from_file(file2, transform_flipped)

    with torch.no_grad():
        features1 = model(image1, return_embeddings=True)
        features2 = model(image2, return_embeddings=True)
        features1_flipped = model(image1_flipped, return_embeddings=True)
        features2_flipped = model(image2_flipped, return_embeddings=True)

    similarity = functional.cosine_similarity(features1+features1_flipped, features2+features2_flipped)
    return similarity.item()
