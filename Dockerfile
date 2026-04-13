FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

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

CMD ["python3", "-m", "src.train"]
