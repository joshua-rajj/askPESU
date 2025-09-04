FROM python:3.11-slim

# system deps (tokenizers need rust sometimes; often fine without)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache the SentenceTransformer model to the image
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("Alibaba-NLP/gte-modernbert-base")
print("Model cached")
PY

ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

COPY . .

# Cloud Run will pass $PORT
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
