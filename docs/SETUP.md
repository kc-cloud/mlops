# Setup Guide - Secure MLOps Pipeline

This guide will walk you through setting up the secure MLOps pipeline from scratch.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [AWS Infrastructure Setup](#aws-infrastructure-setup)
3. [Environment Configuration](#environment-configuration)
4. [Running the Pipeline](#running-the-pipeline)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools
- AWS CLI v2.x or higher
- Python 3.10 or higher
- Docker 20.x or higher
- Git
- Jupyter Notebook (optional)

### AWS Account Requirements
- AWS account with admin access (for initial setup)
- AWS CLI configured with credentials
- Sufficient service quotas for:
  - SageMaker training instances (ml.g5.xlarge or similar)
  - SageMaker endpoints
  - ECR repositories

### Cost Estimate
- **Development/Testing**: ~$50-100/month
- **Production**: ~$300-500/month (depending on usage)

---

## AWS Infrastructure Setup

### Step 1: Deploy Infrastructure with CloudFormation

```bash
# Clone the repository
git clone <repository-url>
cd mlops

# Deploy the CloudFormation stack
aws cloudformation create-stack \
  --stack-name secure-mlops-infra \
  --template-body file://config/cloudformation_template.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --region us-east-1

# Wait for stack creation (10-15 minutes)
aws cloudformation wait stack-create-complete \
  --stack-name secure-mlops-infra \
  --region us-east-1

# Get outputs
aws cloudformation describe-stacks \
  --stack-name secure-mlops-infra \
  --query 'Stacks[0].Outputs' \
  --output table
```

### Step 2: Store Secrets

Store your HuggingFace token in AWS Secrets Manager:

```bash
# Interactive prompt for token
read -s HUGGINGFACE_TOKEN
echo "Token entered (hidden)"

# Create secret
aws secretsmanager create-secret \
  --name huggingface/api-token \
  --description "HuggingFace API token for model downloads" \
  --secret-string "$HUGGINGFACE_TOKEN" \
  --region us-east-1

# Verify
aws secretsmanager describe-secret \
  --secret-id huggingface/api-token \
  --region us-east-1
```

To get a HuggingFace token:
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with read access
3. Copy the token

### Step 3: Update Configuration Files

Get the outputs from CloudFormation and update your configuration:

```bash
# Get execution role ARN
ROLE_ARN=$(aws cloudformation describe-stacks \
  --stack-name secure-mlops-infra \
  --query 'Stacks[0].Outputs[?OutputKey==`SageMakerExecutionRoleArn`].OutputValue' \
  --output text)

# Get KMS key ID
KMS_KEY_ID=$(aws cloudformation describe-stacks \
  --stack-name secure-mlops-infra \
  --query 'Stacks[0].Outputs[?OutputKey==`KMSKeyId`].OutputValue' \
  --output text)

# Get security group ID
SG_ID=$(aws cloudformation describe-stacks \
  --stack-name secure-mlops-infra \
  --query 'Stacks[0].Outputs[?OutputKey==`SecurityGroupId`].OutputValue' \
  --output text)

# Get subnet IDs
SUBNET_1=$(aws cloudformation describe-stacks \
  --stack-name secure-mlops-infra \
  --query 'Stacks[0].Outputs[?OutputKey==`Subnet1Id`].OutputValue' \
  --output text)

SUBNET_2=$(aws cloudformation describe-stacks \
  --stack-name secure-mlops-infra \
  --query 'Stacks[0].Outputs[?OutputKey==`Subnet2Id`].OutputValue' \
  --output text)

# Get S3 bucket
BUCKET=$(aws cloudformation describe-stacks \
  --stack-name secure-mlops-infra \
  --query 'Stacks[0].Outputs[?OutputKey==`S3BucketName`].OutputValue' \
  --output text)

echo "Role ARN: $ROLE_ARN"
echo "KMS Key: $KMS_KEY_ID"
echo "Security Group: $SG_ID"
echo "Subnets: $SUBNET_1, $SUBNET_2"
echo "Bucket: $BUCKET"
```

Update `config/security_config.yaml`:

```yaml
security:
  iam:
    execution_role: "arn:aws:iam::ACCOUNT_ID:role/..." # Use $ROLE_ARN

  encryption:
    s3_kms_key_id: "arn:aws:kms:..." # Use $KMS_KEY_ID
    volume_kms_key_id: "arn:aws:kms:..." # Use $KMS_KEY_ID

  vpc:
    security_group_ids:
      - "sg-xxxxx"  # Use $SG_ID
    subnets:
      - "subnet-xxxxx"  # Use $SUBNET_1
      - "subnet-yyyyy"  # Use $SUBNET_2
```

---

## Environment Configuration

### Step 1: Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import sagemaker; print(f'SageMaker SDK: {sagemaker.__version__}')"
```

### Step 2: Configure AWS CLI

```bash
# Configure AWS CLI if not already done
aws configure

# Verify configuration
aws sts get-caller-identity

# Set default region
export AWS_DEFAULT_REGION=us-east-1
```

### Step 3: Install Docker (if not already installed)

**macOS**:
```bash
brew install --cask docker
```

**Ubuntu**:
```bash
sudo apt-get update
sudo apt-get install docker.io
sudo systemctl start docker
sudo usermod -aG docker $USER
```

**Windows**: Download Docker Desktop from https://docker.com

### Step 4: (Optional) Install Trivy for Local Scanning

```bash
# macOS
brew install trivy

# Ubuntu
wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
sudo apt-get update
sudo apt-get install trivy

# Verify
trivy --version
```

---

## Running the Pipeline

### Option 1: Run via Jupyter Notebook (Recommended for First Run)

```bash
# Start Jupyter
jupyter notebook notebooks/secure_mlops_pipeline.ipynb

# Follow the notebook step-by-step
# Each cell is documented and can be run independently
```

### Option 2: Run Individual Components

#### 1. Download Model Securely

```bash
python -c "
from src.model_management.secure_model_downloader import SecureModelDownloader

downloader = SecureModelDownloader()
model_path = downloader.download_model(
    model_id='gpt2',
    local_dir='./models/base-model'
)
print(f'Model downloaded to: {model_path}')
"
```

#### 2. Build and Push Container

```bash
# Make script executable
chmod +x scripts/build_and_push.sh

# Build and push
./scripts/build_and_push.sh secure-mlops-training v1.0

# Check image URI
cat .ecr_image_uri
```

#### 3. Prepare Training Data

```bash
# Create sample dataset (or use your own)
python -c "
from datasets import load_dataset

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:1000]')
dataset.save_to_disk('./data/train')

eval_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation[:100]')
eval_dataset.save_to_disk('./data/validation')

print('Dataset prepared')
"

# Upload to S3
BUCKET=$(aws cloudformation describe-stacks \
  --stack-name secure-mlops-infra \
  --query 'Stacks[0].Outputs[?OutputKey==`S3BucketName`].OutputValue' \
  --output text)

aws s3 sync ./data/train/ s3://$BUCKET/data/train/ --sse aws:kms
aws s3 sync ./data/validation/ s3://$BUCKET/data/validation/ --sse aws:kms
```

#### 4. Run Training

See the Jupyter notebook for the complete training flow with SageMaker.

---

## Verification Steps

### 1. Verify ECR Repository

```bash
aws ecr describe-repositories \
  --repository-names secure-mlops-training \
  --region us-east-1
```

### 2. Verify Secrets

```bash
aws secretsmanager describe-secret \
  --secret-id huggingface/api-token \
  --region us-east-1
```

### 3. Verify S3 Bucket

```bash
aws s3 ls s3://$BUCKET/
```

### 4. Verify IAM Role

```bash
aws iam get-role --role-name secure-mlops-sagemaker-execution-role
```

### 5. Test SageMaker Access

```bash
python -c "
import sagemaker
session = sagemaker.Session()
print(f'SageMaker session initialized')
print(f'Default bucket: {session.default_bucket()}')
print(f'Region: {session.boto_region_name}')
"
```

---

## Troubleshooting

### Issue 1: "Access Denied" when accessing Secrets Manager

**Solution**:
```bash
# Add permission to your IAM user/role
aws iam attach-user-policy \
  --user-name YOUR_USERNAME \
  --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite
```

### Issue 2: Docker build fails

**Solution**:
```bash
# Check Docker is running
docker ps

# If on macOS/Windows, ensure Docker Desktop is running
# Try building with verbose output
docker build --progress=plain -t test .
```

### Issue 3: ECR push fails with authentication error

**Solution**:
```bash
# Re-authenticate with ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com
```

### Issue 4: SageMaker training job fails with "ResourceLimitExceeded"

**Solution**:
```bash
# Check service quotas
aws service-quotas get-service-quota \
  --service-code sagemaker \
  --quota-code L-1234567890

# Request quota increase through AWS Console
# Or use a smaller instance type: ml.m5.xlarge
```

### Issue 5: KMS key access denied

**Solution**:
```bash
# Update KMS key policy to grant access
# Get the key policy
aws kms get-key-policy \
  --key-id $KMS_KEY_ID \
  --policy-name default \
  --output text > key-policy.json

# Edit key-policy.json to add your principal
# Update the policy
aws kms put-key-policy \
  --key-id $KMS_KEY_ID \
  --policy-name default \
  --policy file://key-policy.json
```

### Issue 6: VPC connectivity issues

**Solution**:
- Ensure VPC endpoints are created for S3, SageMaker, ECR
- Check security group rules allow outbound HTTPS (443)
- Verify subnets have route to VPC endpoints

```bash
# List VPC endpoints
aws ec2 describe-vpc-endpoints \
  --filters "Name=vpc-id,Values=YOUR_VPC_ID"
```

---

## Next Steps

1. **Run the demo pipeline**: Start with the Jupyter notebook
2. **Customize for your use case**: Modify configs and data
3. **Set up monitoring**: Configure CloudWatch alarms
4. **Review security**: Run security audit with AWS Security Hub
5. **Document**: Add your own documentation for team

---

## Additional Resources

### AWS Documentation
- [SageMaker Security](https://docs.aws.amazon.com/sagemaker/latest/dg/security.html)
- [ECR Security Best Practices](https://docs.aws.amazon.com/AmazonECR/latest/userguide/security-best-practices.html)
- [KMS Best Practices](https://docs.aws.amazon.com/kms/latest/developerguide/best-practices.html)

### Training Resources
- [SageMaker Examples](https://github.com/aws/amazon-sagemaker-examples)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [MLOps Best Practices](https://ml-ops.org/)

### Support
- GitHub Issues: <repository-url>/issues
- AWS Support: https://console.aws.amazon.com/support/

---

## Cleanup

To remove all resources and avoid charges:

```bash
# Delete SageMaker endpoints
aws sagemaker delete-endpoint --endpoint-name secure-llm-endpoint

# Delete ECR images
aws ecr batch-delete-image \
  --repository-name secure-mlops-training \
  --image-ids imageTag=v1.0

# Delete CloudFormation stack (this deletes most resources)
aws cloudformation delete-stack \
  --stack-name secure-mlops-infra

# Delete Secrets Manager secret
aws secretsmanager delete-secret \
  --secret-id huggingface/api-token \
  --force-delete-without-recovery

# Empty and delete S3 bucket
BUCKET=$(aws cloudformation describe-stacks \
  --stack-name secure-mlops-infra \
  --query 'Stacks[0].Outputs[?OutputKey==`S3BucketName`].OutputValue' \
  --output text)
aws s3 rm s3://$BUCKET --recursive
aws s3 rb s3://$BUCKET
```

---

**Happy Secure MLOps!**
