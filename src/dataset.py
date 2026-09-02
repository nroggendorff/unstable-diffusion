import itertools

import torch
from torchvision import transforms
from PIL import Image

from datasets import load_dataset

DATASET_ID = "none-yet/processed-anime"
SHUFFLE_BUFFER = 1000

BUCKETS = [
    (512, 512),
    (512, 448),
    (448, 512),
    (512, 384),
    (384, 512),
    (512, 320),
    (320, 512),
    (512, 256),
    (256, 512),
]

_NORMALIZE = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def assign_bucket(width: int, height: int) -> tuple[int, int]:
    aspect = width / height
    return min(BUCKETS, key=lambda b: abs(b[0] / b[1] - aspect))


def _cover_resize(image: Image.Image, bucket: tuple[int, int]) -> Image.Image:
    bucket_w, bucket_h = bucket
    width, height = image.size
    scale = max(bucket_w / width, bucket_h / height)
    resized = (
        max(bucket_w, int(round(width * scale))),
        max(bucket_h, int(round(height * scale))),
    )
    return image.resize(resized, Image.Resampling.LANCZOS)


def shuffle_buffer_size(n: int, shuffle_buffer: int = SHUFFLE_BUFFER) -> int:
    return min(shuffle_buffer, max(n, 256))


def get_samples(n=100, seed=0, shuffle_buffer=SHUFFLE_BUFFER):
    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    dataset = dataset.shuffle(
        seed=seed, buffer_size=shuffle_buffer_size(n, shuffle_buffer)
    )

    iterator = iter(dataset)
    try:
        yield from itertools.islice(iterator, n)
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


def prepare_sample(sample, device):
    image = sample.get("image", None)
    prompt = sample.get("text", "anime")

    if image is None or not isinstance(image, Image.Image):
        return None, None, None

    image = image.convert("RGB")
    bucket = assign_bucket(*image.size)

    image = _cover_resize(image, bucket)
    bucket_w, bucket_h = bucket
    left = (image.width - bucket_w) // 2
    top = (image.height - bucket_h) // 2
    image = image.crop((left, top, left + bucket_w, top + bucket_h))

    tensor = _NORMALIZE(image).unsqueeze(0).to(device, dtype=torch.float16)
    return tensor, prompt, bucket
