# Docker Build & Push Guide

This guide explains how to build Docker images for the Agentic Forecasting Bootcamp and push them to Google Cloud Artifact Registry.

## Registry Location

- **Project:** `coderd`
- **Location:** `us-central1`
- **Repository:** `coder`
- **Image:** `agentic-forecasting-bootcamp`
- **Full path:** `us-central1-docker.pkg.dev/coderd/coder/agentic-forecasting-bootcamp`

## Prerequisites

- [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed and authenticated
- Access to the `coderd` Google Cloud project

## Setup

### 1. Authenticate with gcloud

```sh
gcloud auth login
gcloud auth application-default login
gcloud config set project coderd
```

### 2. Configure Docker for Artifact Registry

```sh
gcloud auth configure-docker us-central1-docker.pkg.dev
```

## Build and Push

### Option A: Using Cloud Build (recommended)

Build and push directly from the repo root:

```sh
cd /path/to/agentic-forecasting
gcloud builds submit --project=coderd --region=us-central1 \
  --config=deploy/docker/cloudbuild.yaml .
```

### Option B: Build locally and push

```sh
cd /path/to/agentic-forecasting

# Build the image
docker build -t us-central1-docker.pkg.dev/coderd/coder/agentic-forecasting-bootcamp:latest \
  -f deploy/docker/Dockerfile .

# Push to Artifact Registry
docker push us-central1-docker.pkg.dev/coderd/coder/agentic-forecasting-bootcamp:latest
```

## Managing Images

### List existing images

```sh
gcloud artifacts docker images list \
  us-central1-docker.pkg.dev/coderd/coder/agentic-forecasting-bootcamp \
  --include-tags
```

### Delete an image (if needed)

```sh
gcloud artifacts docker images delete \
  us-central1-docker.pkg.dev/coderd/coder/agentic-forecasting-bootcamp:latest \
  --quiet
```

### Pull the image

```sh
docker pull us-central1-docker.pkg.dev/coderd/coder/agentic-forecasting-bootcamp:latest
```

## References

- [Build and push a Docker image with Cloud Build](https://cloud.google.com/build/docs/build-push-docker-image)
- [Artifact Registry Docker Guide](https://cloud.google.com/artifact-registry/docs/docker)
- [gcloud CLI Documentation](https://cloud.google.com/sdk/gcloud)
