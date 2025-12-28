# Docker Setup Guide

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Build and run the container
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Using Docker CLI

```bash
# Build the image
docker build -t breast-cancer-api:latest .

# Run the container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  breast-cancer-api:latest

# Run in background
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  --name breast-cancer-api \
  breast-cancer-api:latest
```

## Access the API

Once the container is running:

- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Docker Details

### Image Information
- **Base Image**: `python:3.12-slim`
- **Working Directory**: `/app`
- **Exposed Port**: `8000`
- **Default Command**: `uvicorn main:app --host 0.0.0.0 --port 8000`

### Environment Variables
- `PYTHONUNBUFFERED=1` - Unbuffered Python output
- `PYTHONDONTWRITEBYTECODE=1` - Don't write .pyc files
- `PIP_NO_CACHE_DIR=1` - Don't cache pip packages

### Health Check

The container includes a built-in health check that:
- Runs every 30 seconds
- Times out after 10 seconds
- Requires 3 consecutive failures to mark unhealthy
- Waits 40 seconds before first check

Check container health:
```bash
docker ps
# Look at STATUS column - should show "healthy" if working
```

## Volume Mounts

The container uses volumes to persist data:

```
/app/models  <- Model pickle files
/app/data    <- CSV data files
```

These are mounted from your local directories, so model training/updates are saved.

## Build Customization

### Building with custom tag
```bash
docker build -t myregistry/breast-cancer-api:v1.0.0 .
```

### Building for specific architecture
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t myregistry/breast-cancer-api .
```

## Running Tests in Docker

```bash
# Run with test dependencies
docker run -it breast-cancer-api:latest pytest
```

## Pushing to Registry

### Docker Hub
```bash
# Tag the image
docker tag breast-cancer-api:latest username/breast-cancer-api:latest

# Push to Docker Hub
docker push username/breast-cancer-api:latest
```

### GitHub Container Registry
```bash
# Tag the image
docker tag breast-cancer-api:latest ghcr.io/username/breast-cancer-api:latest

# Push to GHCR
docker push ghcr.io/username/breast-cancer-api:latest
```

## Troubleshooting

### Container exits immediately
```bash
# Check logs
docker logs breast-cancer-api

# Run with interactive terminal
docker run -it breast-cancer-api:latest bash
```

### Port already in use
```bash
# Use a different port
docker run -p 9000:8000 breast-cancer-api:latest

# Find process using port
lsof -i :8000
```

### Model or data not found
```bash
# Ensure volumes are mounted correctly
docker run -v $(pwd)/models:/app/models \
           -v $(pwd)/data:/app/data \
           breast-cancer-api:latest
```

### Permission issues with volumes
```bash
# Run with specific user ID
docker run --user 1000:1000 \
           -v $(pwd)/models:/app/models \
           breast-cancer-api:latest
```

## Security Best Practices

1. **Use specific Python version**: `python:3.12-slim` (not `latest`)
2. **Minimize image size**: Using `-slim` variant
3. **Don't run as root**: Consider adding non-root user in production
4. **Security scanning**: Use tools like Trivy or Snyk
5. **Private registry**: Use private Docker registries for sensitive images

Example with non-root user:
```dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

## Production Deployment

### With Gunicorn (production-grade)
```dockerfile
RUN pip install gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "main:app"]
```

### With environment-specific configs
```bash
docker run -e PORT=8000 \
           -e LOG_LEVEL=info \
           breast-cancer-api:latest
```

## Monitoring

### Container metrics
```bash
# View resource usage
docker stats breast-cancer-api

# View detailed info
docker inspect breast-cancer-api
```

## Cleanup

```bash
# Remove container
docker rm breast-cancer-api

# Remove image
docker rmi breast-cancer-api:latest

# Remove unused images, containers, networks
docker system prune

# Full cleanup (use with caution)
docker system prune -a
```

## Kubernetes Deployment

Example deployment.yaml:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: breast-cancer-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: breast-cancer-api
  template:
    metadata:
      labels:
        app: breast-cancer-api
    spec:
      containers:
      - name: api
        image: breast-cancer-api:latest
        ports:
        - containerPort: 8000
        healthCheck:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 40
          periodSeconds: 30
```
