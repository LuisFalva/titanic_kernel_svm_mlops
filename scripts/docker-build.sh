#!/bin/bash -e

# Define the image name and tag
image="titanic_kernel_svm_mlops"
tag=${1:-"latest"}

# Prune and freeing up memory
docker system prune -a -f --volumes

# Build the Docker image
docker build -t "$image":"$tag" .

# Tag the image with the registry name
registry_image_name="luisfalva/$image"
docker tag "$image":"$tag" "$registry_image_name":"$tag"
