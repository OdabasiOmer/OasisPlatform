#!/bin/bash

# Important: Change this to model directory since it overwrites Dockerfile env variable!
export OASIS_MODEL_DATA_DIR=/home/omer/GitHub/OasisRed/

# Note: no need to redo api_server since we dont touch it
#docker rmi coreoasis/api_server:latest
docker rmi coreoasis/model_worker:latest
docker rmi coreoasis/model_worker:1.28.0

# Note: no need to redo api_server since we dont touch it
#docker build -f Dockerfile.api_server -t coreoasis/api_server .

# Note: Commenting this out because the worker container is built from inside the .yml based on Dockerfile.model_worker file.
#docker build -f Dockerfile.model_worker -t coreoasis/model_worker:1.28.0 .

docker-compose -f docker-compose.yml -f oasisui-docker-standalone.yml up -d
