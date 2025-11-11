# Secure Dockerfile for SageMaker Training
# Base image from AWS Deep Learning Containers (regularly scanned and updated)
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu118-ubuntu20.04-sagemaker

# Metadata
LABEL maintainer="mlops-security-team"
LABEL version="1.0"
LABEL description="Secure container for LLM fine-tuning"

# Security: Run as non-root user
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} mlopsuser && \
    useradd -m -u ${USER_ID} -g mlopsuser mlopsuser

# Set working directory
WORKDIR /opt/ml/code

# Security: Update packages and remove unnecessary tools
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies with hash checking
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip check

# Copy application code
COPY src/ /opt/ml/code/src/
COPY config/ /opt/ml/code/config/

# Security: Set proper permissions
RUN chown -R mlopsuser:mlopsuser /opt/ml/code && \
    chmod -R 755 /opt/ml/code

# Security: Remove write permissions from sensitive files
RUN chmod 444 /opt/ml/code/requirements.txt

# Create necessary directories with proper permissions
RUN mkdir -p /opt/ml/model /opt/ml/output /opt/ml/input && \
    chown -R mlopsuser:mlopsuser /opt/ml

# Security: Switch to non-root user
USER mlopsuser

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=1

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/opt/ml/code"

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; import transformers; print('OK')"

# Entry point for SageMaker
ENTRYPOINT ["python", "src/training/train.py"]
