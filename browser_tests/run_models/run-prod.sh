#!/bin/bash
# Temporarily comment out to test without exiting shell:
# set -euo pipefail 

cd "$(dirname "$0")"

set -a
[ -f ./.env ] && source ./.env
set +a

if [ -z "${DOCKER_PLATFORM:-}" ]; then
    docker_arch="$(docker version --format '{{.Server.Arch}}' 2>/dev/null || uname -m)"
    case "$docker_arch" in
        arm64|aarch64)
            DOCKER_PLATFORM=linux/arm64
            ;;
        *)
            DOCKER_PLATFORM=linux/amd64
            ;;
    esac
fi

docker build --platform="$DOCKER_PLATFORM" --tag 'koel-api' -f ./Dockerfile ../..
docker run -t -i -p 8080:8080 \
    -e API_KEY \
    -e AWS_BUCKET_NAME \
    -e AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY \
    -e GOOGLE_PROJECT_ID \
    -e GOOGLE_APPLICATION_CREDENTIALS \
    -e GOOGLE_BUCKET_NAME \
    -e HF_TOKEN \
    'koel-api'
