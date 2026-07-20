FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for PaddleOCR and OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libwebp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Create a non-root user to run the app as
RUN groupadd --system app && useradd --system --gid app --home /app app

# Create uploads directory inside container (will be mounted)
RUN mkdir -p /app/uploads

COPY . .
RUN chown -R app:app /app

USER app

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
