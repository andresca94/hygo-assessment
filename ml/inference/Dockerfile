FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY ml/inference/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

CMD ["uvicorn", "ml.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
