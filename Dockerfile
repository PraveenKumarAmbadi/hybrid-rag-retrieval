FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install Python and pip (no venv needed inside Docker)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip and install dependencies with the custom PyTorch index
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130

# Copy the rest of the application code
COPY . .

# Document the intended port
EXPOSE 8000

# Run the FastAPI app, bound to all interfaces
CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8000"]