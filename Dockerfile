FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# CPU-only torch first (the default PyPI wheel pulls CUDA, ~6 GB); pip then sees torch satisfied.
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml ./
COPY prahari ./prahari
RUN pip install --no-cache-dir -e .
COPY . .
EXPOSE 8000
CMD ["uvicorn", "prahari.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
