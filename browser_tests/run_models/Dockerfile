FROM python:3.10.19-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    espeak-ng \
    libportaudio2 \
    python3-pyaudio \
 && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY browser_tests/run_models/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the files
COPY models/ models/
COPY scripts/ scripts/
COPY browser_tests/run_models/main.py main.py

EXPOSE 8080

CMD ["gunicorn", "--workers=2", "--threads=1", "--bind=0.0.0.0:8080", "--timeout", "6000", "main:app"]
