FROM public.ecr.aws/deep-learning-containers/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    libsndfile1 \
    lilypond \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src /app/src/

ENTRYPOINT ["src/entrypoint.sh"]
CMD ["train"]
