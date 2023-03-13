#!/bin/bash -e

# Define the image name and tag
image="titanic_kernel_svm_mlops"
tag=${1:-"latest"}

# Build the Docker image
echo docker build -t "$image":"$tag" .

# Tag the image with the registry name
registry_image_name="luisfalva/$image"
echo docker tag "$image":"$tag" "$registry_image_name":"$tag"
