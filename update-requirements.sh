#!/bin/bash
# Helper script to update python requirment files within the docker container 

docker build --build-arg DOCKER_USER=$(whoami) -f docker/Dockerfile.update_python_req -t coreoasis/platform_req:update .

# run as mapped user 
USER="$(id -u):$(id -g)"

# run as root 
#USER='root'

echo 'Update requirements-server.txt'
docker run --user $USER -v $PWD:/mnt -w /mnt coreoasis/platform_req:update  /bin/sh -c "pip-compile requirements-server.in"

echo 'Update requirements-worker.txt'
docker run --user $USER -v $PWD:/mnt -w /mnt coreoasis/platform_req:update  /bin/sh -c "pip-compile requirements-worker.in"

echo 'Update requirements.txt'
docker run --user $USER -v $PWD:/mnt -w /mnt coreoasis/platform_req:update  /bin/sh -c "pip-compile requirements.in"
