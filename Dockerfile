FROM python:3.13-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg2 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-12-6 \
        libcublas-12-6 \
        libcusparse-12-6 \
        libcurand-12-6 \
        libcusolver-12-6 \
        build-essential \
        curl \
        dirmngr \
        lsb-release \
        fontconfig \
        libfontconfig1-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1
RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

COPY ./requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 5050

CMD ["python", "app.py"]
