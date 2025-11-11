#!/bin/bash
# Secure script to build and push Docker image to ECR
# Usage: ./build_and_push.sh [repository-name] [image-tag]

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Configuration
REPOSITORY_NAME=${1:-"secure-mlops-training"}
IMAGE_TAG=${2:-"latest"}
AWS_REGION=${AWS_REGION:-"us-east-1"}
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting secure Docker build and push process${NC}"
echo "Repository: $REPOSITORY_NAME"
echo "Tag: $IMAGE_TAG"
echo "Region: $AWS_REGION"
echo "Account: $AWS_ACCOUNT_ID"

# Security check: Verify AWS credentials
echo -e "\n${YELLOW}Verifying AWS credentials...${NC}"
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}Error: AWS credentials not configured${NC}"
    exit 1
fi
echo -e "${GREEN}✓ AWS credentials verified${NC}"

# Create ECR repository if it doesn't exist
echo -e "\n${YELLOW}Checking ECR repository...${NC}"
if ! aws ecr describe-repositories --repository-names "$REPOSITORY_NAME" --region "$AWS_REGION" &> /dev/null; then
    echo "Creating ECR repository with security settings..."
    aws ecr create-repository \
        --repository-name "$REPOSITORY_NAME" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=KMS \
        --image-tag-mutability IMMUTABLE \
        --region "$AWS_REGION"

    # Enable enhanced scanning
    aws ecr put-registry-scanning-configuration \
        --scan-type ENHANCED \
        --rules '[{"scanFrequency":"CONTINUOUS_SCAN","repositoryFilters":[{"filter":"*","filterType":"WILDCARD"}]}]' \
        --region "$AWS_REGION" || echo "Enhanced scanning may not be available in this region"
fi
echo -e "${GREEN}✓ ECR repository ready${NC}"

# Docker login to ECR
echo -e "\n${YELLOW}Logging in to ECR...${NC}"
aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin \
    "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
echo -e "${GREEN}✓ Logged in to ECR${NC}"

# Build Docker image
echo -e "\n${YELLOW}Building Docker image...${NC}"
IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPOSITORY_NAME:$IMAGE_TAG"

docker build \
    --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
    --build-arg VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown") \
    --tag "$IMAGE_URI" \
    --file Dockerfile \
    .

echo -e "${GREEN}✓ Docker image built successfully${NC}"

# Optional: Run local security scan with Trivy (if available)
echo -e "\n${YELLOW}Running local security scan...${NC}"
if command -v trivy &> /dev/null; then
    trivy image --severity HIGH,CRITICAL "$IMAGE_URI" || echo "Trivy scan completed with findings"
else
    echo "Trivy not installed. Install with: brew install trivy"
    echo "Skipping local scan. ECR will scan on push."
fi

# Push image to ECR
echo -e "\n${YELLOW}Pushing image to ECR...${NC}"
docker push "$IMAGE_URI"
echo -e "${GREEN}✓ Image pushed successfully${NC}"

# Tag as latest (optional)
if [ "$IMAGE_TAG" != "latest" ]; then
    LATEST_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPOSITORY_NAME:latest"
    docker tag "$IMAGE_URI" "$LATEST_URI"
    docker push "$LATEST_URI"
    echo -e "${GREEN}✓ Also tagged and pushed as latest${NC}"
fi

# Wait for ECR scan results
echo -e "\n${YELLOW}Waiting for ECR vulnerability scan...${NC}"
sleep 10  # Give ECR time to start the scan

for i in {1..30}; do
    SCAN_STATUS=$(aws ecr describe-image-scan-findings \
        --repository-name "$REPOSITORY_NAME" \
        --image-id imageTag="$IMAGE_TAG" \
        --region "$AWS_REGION" \
        --query 'imageScanStatus.status' \
        --output text 2>/dev/null || echo "IN_PROGRESS")

    if [ "$SCAN_STATUS" = "COMPLETE" ]; then
        echo -e "${GREEN}✓ Scan completed${NC}"

        # Get scan findings
        aws ecr describe-image-scan-findings \
            --repository-name "$REPOSITORY_NAME" \
            --image-id imageTag="$IMAGE_TAG" \
            --region "$AWS_REGION" \
            --query 'imageScanFindings.findingSeverityCounts' \
            --output table

        # Check for critical vulnerabilities
        CRITICAL=$(aws ecr describe-image-scan-findings \
            --repository-name "$REPOSITORY_NAME" \
            --image-id imageTag="$IMAGE_TAG" \
            --region "$AWS_REGION" \
            --query 'imageScanFindings.findingSeverityCounts.CRITICAL' \
            --output text 2>/dev/null || echo "0")

        if [ "$CRITICAL" != "None" ] && [ "$CRITICAL" -gt 0 ]; then
            echo -e "${RED}⚠ WARNING: Found $CRITICAL CRITICAL vulnerabilities${NC}"
            echo "Review findings before deploying to production"
        fi
        break
    elif [ "$SCAN_STATUS" = "FAILED" ]; then
        echo -e "${RED}✗ Scan failed${NC}"
        break
    else
        echo "Scan status: $SCAN_STATUS (attempt $i/30)"
        sleep 10
    fi
done

# Output image URI
echo -e "\n${GREEN}====================================${NC}"
echo -e "${GREEN}Build and push completed successfully${NC}"
echo -e "${GREEN}====================================${NC}"
echo "Image URI: $IMAGE_URI"
echo ""
echo "To use this image in SageMaker:"
echo "  image_uri = '$IMAGE_URI'"

# Save image URI to file for use in other scripts
echo "$IMAGE_URI" > .ecr_image_uri
echo -e "\n${GREEN}Image URI saved to .ecr_image_uri${NC}"
