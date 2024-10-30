FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/nlp

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade \
    pip==24.3.1 \
    setuptools==75.3.0 \
    wheel==0.44.0

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir \
    numpy==2.0.2 \
    pandas==2.2.3 \
    scipy==1.14.1

RUN pip install --no-cache-dir \
    torch==2.5.1 \
    tensorflow==2.18.0 \
    scikit-learn==1.5.2

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

RUN mkdir -p /app/nlp/models \
    /app/nlp/cache \
    /app/nlp/data \
    /app/nlp/logs

ENV PYTHONPATH=/app
ENV TORCH_HOME=/app/nlp/cache/torch 
ENV TRANSFORMERS_CACHE=/app/nlp/cache/transformers
ENV HF_HOME=/app/nlp/cache/huggingface

COPY . .

RUN useradd -m -u 1000 nlpuser
RUN chown -R nlpuser:nlpuser /app
USER nlpuser 

EXPOSE 8888 8000

CMD ["/bin/bash"]
