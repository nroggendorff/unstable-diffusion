import torch
from torchvision import transforms
from PIL import Image

from datasets import load_dataset


DATASET_ID = "none-yet/anime-captions"
IMAGE_SIZE = 1024


def get_samples(n=100):
    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    out = []
    for i, x in enumerate(dataset):
        out.append(x)
        if i >= n:
            break
    return out


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def prepare_sample(sample, transform, device):
    image = sample.get("image", None)
    prompt = sample.get("text", "anime")

    if image is None or not isinstance(image, Image.Image):
        return None, None

    image = transform(image).unsqueeze(0).to(device, dtype=torch.float16)
    return image, prompt
