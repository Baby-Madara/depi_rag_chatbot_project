FROM python:3.10-slim

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Use a higher timeout and no progress bar to prevent hanging
#RUN pip install --no-cache-dir --default-timeout=100 --progress-bar off -r requirements.txt
# Force the lightweight CPU version of PyTorch first, and don't save the installer cache
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Then install the rest of your requirements
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]