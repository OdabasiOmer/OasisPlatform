#!/bin/bash

# Important: Set this variable accordingly: set it to the model directory; it overwrites Dockerfile env variable.
# export OASIS_MODEL_DATA_DIR=/data/OasisRed/

# Note: no need to redo api_server since we dont touch it
#docker rmi coreoasis/api_server:latest
docker rmi coreoasis/model_worker:latest
docker rmi coreoasis/model_worker:1.26.2

# Note: no need to redo api_server since we dont touch it
#docker build -f Dockerfile.api_server -t coreoasis/api_server .

# Note: Commenting this out because the worker container is built from inside the .yml 
# based on Dockerfile.model_worker file, which is fine.
# docker build -f Dockerfile.model_worker -t coreoasis/model_worker:1.26.2 .

docker-compose -f docker-compose.yml -f oasisui-docker-standalone.yml up -d
docker-compose -f portainer.yaml up -d
