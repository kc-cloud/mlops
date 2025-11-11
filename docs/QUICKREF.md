# Quick Reference Guide

Essential commands and code snippets for the Secure MLOps Pipeline.

## Common Commands

### AWS Setup

```bash
# Deploy infrastructure
aws cloudformation create-stack \
  --stack-name secure-mlops-infra \
  --template-body file://config/cloudformation_template.yaml \
  --capabilities CAPABILITY_NAMED_IAM

# Store HuggingFace token
aws secretsmanager create-secret \
  --name huggingface/api-token \
  --secret-string "hf_xxxxx"

# Get stack outputs
aws cloudformation describe-stacks \
  --stack-name secure-mlops-infra \
  --query 'Stacks[0].Outputs' \
  --output table
```

### Container Management

```bash
# Build and push to ECR
./scripts/build_and_push.sh secure-mlops-training v1.0

# Get image URI
cat .ecr_image_uri

# Check scan results
aws ecr describe-image-scan-findings \
  --repository-name secure-mlops-training \
  --image-id imageTag=v1.0
```

### Model Operations

```bash
# Download model securely
python -c "
from src.model_management.secure_model_downloader import SecureModelDownloader
downloader = SecureModelDownloader()
model_path = downloader.download_model(model_id='gpt2')
print(f'Downloaded to: {model_path}')
"

# List model versions
python -c "
from src.deployment.model_registry import SecureModelRegistry
registry = SecureModelRegistry()
versions = registry.list_model_versions('secure-llm-models')
for v in versions:
    print(f\"Version {v['version']}: {v['status']}\")
"
```

## Python Code Snippets

### Secure Model Download

```python
from src.model_management.secure_model_downloader import SecureModelDownloader

downloader = SecureModelDownloader()
model_path = downloader.download_model(
    model_id='meta-llama/Llama-2-7b-hf',
    local_dir='./models',
    cache_dir='./cache'
)

# Upload to S3 with encryption
downloader.upload_to_s3(
    local_path=model_path,
    s3_uri='s3://your-bucket/models/',
    encrypt=True
)
```

### ECR Management

```python
from src.security.ecr_manager import SecureECRManager

ecr = SecureECRManager()

# Create repository
repo = ecr.create_secure_repository(
    repository_name='secure-mlops-training',
    scan_on_push=True,
    enable_encryption=True
)

# Get scan results
results = ecr.get_scan_results(
    repository_name='secure-mlops-training',
    image_tag='v1.0'
)
```

### Model Evaluation

```python
from src.training.evaluator import ModelEvaluator

evaluator = ModelEvaluator(model_path='./trained_model')

# Evaluate
metrics = evaluator.evaluate_metrics(eval_dataset)

# Check thresholds
thresholds = {
    'perplexity_max': 20.0,
    'eval_loss_max': 1.5
}
passed, failures = evaluator.check_thresholds(metrics, thresholds)

if passed:
    print("Model meets all thresholds!")
else:
    print(f"Failed: {failures}")
```

### Model Registry

```python
from src.deployment.model_registry import SecureModelRegistry

registry = SecureModelRegistry()

# Create model package group
registry.create_model_package_group('secure-llm-models')

# Register model
model_arn = registry.register_model(
    model_package_group_name='secure-llm-models',
    model_data_url='s3://bucket/model.tar.gz',
    image_uri='account.dkr.ecr.region.amazonaws.com/image:tag',
    model_metrics={'perplexity': 15.2, 'eval_loss': 1.23},
    approval_status='PendingManualApproval'
)

# Approve model
registry.update_approval_status(
    model_package_arn=model_arn,
    approval_status='Approved'
)
```

### Endpoint Deployment

```python
from src.deployment.deploy import SecureEndpointDeployer

deployer = SecureEndpointDeployer()

# Deploy
endpoint_arn = deployer.deploy_model(
    model_package_arn='arn:aws:sagemaker:...',
    endpoint_name='secure-llm-endpoint',
    instance_type='ml.g5.xlarge',
    instance_count=1,
    enable_monitoring=True,
    enable_autoscaling=True
)

# Test endpoint
result = deployer.invoke_endpoint(
    endpoint_name='secure-llm-endpoint',
    payload='{"inputs": "Once upon a time"}'
)
```

## Configuration Files

### Security Config (`config/security_config.yaml`)

```yaml
security:
  encryption:
    s3_kms_key_id: "alias/sagemaker-kms-key"
    volume_kms_key_id: "alias/sagemaker-volume-kms-key"
    enable_network_isolation: true

  vpc:
    security_group_ids: ["sg-xxxxx"]
    subnets: ["subnet-xxxxx", "subnet-yyyyy"]

  iam:
    execution_role: "arn:aws:iam::ACCOUNT:role/SageMakerRole"
```

### Training Config (`config/training_config.yaml`)

```yaml
model:
  base_model: "meta-llama/Llama-2-7b-hf"

training:
  instance_type: "ml.g5.2xlarge"
  epochs: 3
  batch_size: 4
  learning_rate: 2.0e-5

evaluation:
  performance_threshold:
    perplexity_max: 20.0
    loss_max: 1.5
```

## Monitoring

### CloudWatch Logs

```bash
# View training logs
aws logs tail /aws/sagemaker/TrainingJobs --follow

# View endpoint logs
aws logs tail /aws/sagemaker/Endpoints/secure-llm-endpoint --follow

# View security audit logs
aws logs tail /aws/mlops/secure-pipeline --follow
```

### Metrics

```bash
# Get endpoint metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/SageMaker \
  --metric-name ModelLatency \
  --dimensions Name=EndpointName,Value=secure-llm-endpoint \
  --start-time 2025-01-01T00:00:00Z \
  --end-time 2025-01-02T00:00:00Z \
  --period 3600 \
  --statistics Average
```

## Security Checks

### Verify Encryption

```bash
# Check S3 bucket encryption
aws s3api get-bucket-encryption --bucket your-bucket

# Check KMS key
aws kms describe-key --key-id alias/sagemaker-kms-key

# Check ECR encryption
aws ecr describe-repositories \
  --repository-names secure-mlops-training \
  --query 'repositories[0].encryptionConfiguration'
```

### Access Logs

```bash
# View CloudTrail events
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=ResourceType,AttributeValue=AWS::SageMaker::TrainingJob \
  --max-items 10

# View access logs
aws logs filter-log-events \
  --log-group-name /aws/mlops/secure-pipeline \
  --filter-pattern "FAILED"
```

## Troubleshooting

### Check Service Status

```bash
# SageMaker training job
aws sagemaker describe-training-job --training-job-name job-name

# Endpoint status
aws sagemaker describe-endpoint --endpoint-name secure-llm-endpoint

# ECR image scan status
aws ecr describe-image-scan-findings \
  --repository-name secure-mlops-training \
  --image-id imageTag=v1.0 \
  --query 'imageScanStatus.status'
```

### Common Issues

**Issue: "Access Denied" errors**
```bash
# Check IAM role permissions
aws iam get-role --role-name SageMakerExecutionRole
aws iam list-attached-role-policies --role-name SageMakerExecutionRole
```

**Issue: Training job fails**
```bash
# Get failure reason
aws sagemaker describe-training-job \
  --training-job-name job-name \
  --query 'FailureReason'

# View logs
aws logs tail /aws/sagemaker/TrainingJobs/job-name --follow
```

**Issue: Endpoint deployment fails**
```bash
# Get endpoint status
aws sagemaker describe-endpoint \
  --endpoint-name secure-llm-endpoint \
  --query '[EndpointStatus, FailureReason]'
```

## Cleanup

```bash
# Delete endpoint
aws sagemaker delete-endpoint --endpoint-name secure-llm-endpoint

# Delete model
aws sagemaker delete-model --model-name model-name

# Delete CloudFormation stack
aws cloudformation delete-stack --stack-name secure-mlops-infra

# Delete ECR images
aws ecr batch-delete-image \
  --repository-name secure-mlops-training \
  --image-ids imageTag=v1.0

# Delete secrets
aws secretsmanager delete-secret \
  --secret-id huggingface/api-token \
  --force-delete-without-recovery
```

## Performance Benchmarks

| Operation | Time | Cost |
|-----------|------|------|
| Model Download | 5-10 min | $0 |
| Container Build | 10-15 min | $0.50 |
| Training (1 epoch, GPT-2) | 30-60 min | $2-5 |
| Evaluation | 5-10 min | $0 |
| Deployment | 10-15 min | $0 |

## Key Metrics Thresholds

| Metric | Good | Excellent | Our Threshold |
|--------|------|-----------|---------------|
| Perplexity | < 20 | < 10 | < 20 |
| Eval Loss | < 1.5 | < 1.0 | < 1.5 |
| Training Time | Optimized | GPU Efficient | Spot Instances |

## Resources

- **Setup Guide**: `docs/SETUP.md`
- **Presentation**: `presentation/SECURE_MLOPS_PRESENTATION.md`
- **Notebook**: `notebooks/secure_mlops_pipeline.ipynb`
- **AWS Docs**: https://docs.aws.amazon.com/sagemaker/
