FROM pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime

RUN pip install --no-cache-dir --break-system-packages \
    "diffusers>=0.40" \
    peft \
    bitsandbytes \
    tqdm \
    pillow \
    datasets \
    transformers \
    safetensors

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app/

COPY src /app/src/

ENTRYPOINT ["src/entrypoint.sh"]
CMD ["train"]
