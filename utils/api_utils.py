import torch
from torchvision import transforms as transforms
import torch.nn.functional as functional
from PIL import Image
from dal import DAL
from data.meta import AGEDB, WANDB


def load_image_from_file(file, transform):
    image = Image.open(file).convert('RGB')
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image


def process_images_and_compute_similarity(file1, file2):
    transform = transforms.Compose([
        transforms.Resize((112, 96)),
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

    with torch.no_grad():
        features1 = model(image1, return_embeddings=True)
        features2 = model(image2, return_embeddings=True)

    similarity = functional.cosine_similarity(features1, features2)
    return similarity.item()
