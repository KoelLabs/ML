#!/bin/bash

docker build --platform=linux/amd64 --tag 'koel-api' -f ./Dockerfile ../..
docker run -t -i -p 8080:8080 'koel-api'
