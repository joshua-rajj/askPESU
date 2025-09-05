# Dockerfile
FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# cache locations
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache model at build time to reduce startup time
# (This downloads weights into /root/.cache/huggingface)
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
print("Pre-downloading model: Alibaba-NLP/gte-modernbert-base ...")
SentenceTransformer("Alibaba-NLP/gte-modernbert-base")
print("Done model cache.")
PY

# Copy app code
COPY main.py .

# Expose the port Hugging Face Spaces expects (7860)
EXPOSE 7860

# Run Uvicorn on port 7860 (HF Spaces expects you to bind here)
CMD ["python", "main.py"]
