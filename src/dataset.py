import itertools
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch
from torchvision import transforms
from PIL import Image

from datasets import load_dataset

DATASET_ID = "none-yet/processed-anime"
SHUFFLE_BUFFER = 1024
CACHE_WORKERS = 6

BUCKETS = [
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1216, 832),
    (832, 1216),
    (1344, 768),
    (768, 1344),
    (1536, 640),
    (640, 1536),
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


def build_time_ids(
    original_size: tuple[int, int],
    crop_top_left: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[float, ...]:
    return (
        float(original_size[0]),
        float(original_size[1]),
        float(crop_top_left[0]),
        float(crop_top_left[1]),
        float(target_size[0]),
        float(target_size[1]),
    )


def prepare_sample(sample, device):
    image = sample.get("image", None)
    prompt = sample.get("text", "anime")

    if image is None or not isinstance(image, Image.Image):
        return None, None, None, None

    image = image.convert("RGB")
    original_size = (image.height, image.width)
    bucket = assign_bucket(*image.size)

    image = _cover_resize(image, bucket)
    bucket_w, bucket_h = bucket
    left = (image.width - bucket_w) // 2
    top = (image.height - bucket_h) // 2
    image = image.crop((left, top, left + bucket_w, top + bucket_h))

    tensor = _NORMALIZE(image).unsqueeze(0).to(device, dtype=torch.float16)
    time_ids = build_time_ids(original_size, (top, left), (bucket_h, bucket_w))
    return tensor, prompt, bucket, time_ids


def prepared_samples(samples, workers: int = CACHE_WORKERS):
    if workers <= 1:
        for sample in samples:
            yield prepare_sample(sample, "cpu")
        return

    iterator = iter(samples)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque(
            pool.submit(prepare_sample, sample, "cpu")
            for sample in itertools.islice(iterator, workers * 2)
        )
        for sample in iterator:
            yield pending.popleft().result()
            pending.append(pool.submit(prepare_sample, sample, "cpu"))
        while pending:
            yield pending.popleft().result()
