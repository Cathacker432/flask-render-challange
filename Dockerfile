FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (better caching)
COPY requirements.txt .

# Install CPU-only torch first
RUN pip install --no-cache-dir --default-timeout=100 \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the rest
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy rest of app
COPY . .

EXPOSE 5001

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5001", "app:app"]