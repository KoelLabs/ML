FROM python:3.10.19-slim

WORKDIR /app
ENV TF_LITE_DISABLE_XNNPACK=1
ENV STT_BINARY=/opt/coqui-stt/stt
ENV LD_LIBRARY_PATH=/opt/coqui-stt
ARG TARGETARCH

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    espeak-ng \
    libgomp1 \
    libportaudio2 \
    libsox3 \
    python3-pyaudio \
 && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY browser_tests/run_models/requirements.txt requirements.txt
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        grep -Ev '^(stt|torchcodec|omnilingual-asr)==' requirements.txt > requirements-docker.txt; \
    else \
        cp requirements.txt requirements-docker.txt; \
    fi \
 && pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements-docker.txt

RUN if [ "$TARGETARCH" = "arm64" ]; then \
        mkdir -p /opt/coqui-stt \
     && python -c "import tarfile, urllib.request; url='https://github.com/coqui-ai/STT/releases/download/v1.4.0/native_client.tflite.linux.aarch64.tar.xz'; path='/tmp/coqui-stt.tar.xz'; urllib.request.urlretrieve(url, path); tarfile.open(path).extractall('/opt/coqui-stt')" \
     && chmod +x /opt/coqui-stt/stt; \
    fi

# Copy the rest of the files
COPY models/deepspeech models/deepspeech
COPY scripts/ scripts/
COPY browser_tests/run_models/main.py main.py

EXPOSE 8080

CMD ["gunicorn", "--workers=2", "--threads=1", "--bind=0.0.0.0:8080", "--timeout", "6000", "main:app"]
