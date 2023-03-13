#!/bin/bash -e

# Define the image name and tag
IMAGE_NAME="titanic_kernel_svm_mlops"
IMAGE_TAG="latest"

# Build the Docker image
docker build -t "$IMAGE_NAME":"$IMAGE_TAG" .

# Tag the image with the registry name
REGISTRY_IMAGE_NAME="luisfalva/$IMAGE_NAME"
docker tag "$IMAGE_NAME":"$IMAGE_TAG" "$REGISTRY_IMAGE_NAME":"$IMAGE_TAG"

# Push the image to the registry
docker push "$REGISTRY_IMAGE_NAME":"$IMAGE_TAG"
