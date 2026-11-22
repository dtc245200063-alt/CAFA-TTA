FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libglib2.0-0 libgl1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Match the source project's PyTorch/CUDA research environment.
RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.10.0 torchvision==0.25.0

RUN python -m pip install \
    medmnist==3.0.2 \
    numpy==2.5.4 \
    tqdm==4.70.1 \
    pillow==13.0.0

WORKDIR /runner
COPY docker_train.py /runner/docker_train.py
ENTRYPOINT ["python", "-u", "/runner/docker_train.py"]
CMD ["--help"]
