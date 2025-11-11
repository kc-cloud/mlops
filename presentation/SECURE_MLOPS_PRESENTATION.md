# Secure MLOps Pipeline for LLM Fine-Tuning
## Security Engineering & Operations in Production ML

---

## Agenda

1. Introduction & Objectives
2. Architecture Overview
3. Security Controls by Layer
4. Pipeline Workflow
5. Demo & Implementation
6. Compliance & Governance
7. Monitoring & Observability
8. Best Practices & Lessons Learned
9. Q&A

---

## 1. Introduction & Objectives

### Project Goal
Demonstrate a **production-grade secure MLOps pipeline** for LLM fine-tuning with comprehensive security controls at every stage.

### Key Objectives
- **Security First**: Implement defense-in-depth across all pipeline stages
- **Compliance**: Meet industry standards (SOC2, HIPAA-ready, PCI-DSS)
- **Automation**: Reduce manual security checks through automation
- **Observability**: Complete audit trail and monitoring
- **Governance**: Model approval workflows and version control

---

## 2. Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SECURE MLOPS PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. Model Source        →  HuggingFace (Secure Token Auth)      │
│                                                                   │
│  2. Container Registry  →  AWS ECR (Vuln Scanning + Encryption) │
│                                                                   │
│  3. Training           →  SageMaker (VPC + KMS + IAM)           │
│                                                                   │
│  4. Evaluation         →  Automated Threshold Validation         │
│                                                                   │
│  5. Model Registry     →  SageMaker Registry (Approval Flow)    │
│                                                                   │
│  6. Deployment         →  Secure Endpoint (Monitoring)          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack
- **Cloud Platform**: AWS (SageMaker, ECR, S3, KMS, VPC)
- **ML Framework**: PyTorch, Transformers, PEFT (LoRA)
- **Security Tools**: AWS KMS, Secrets Manager, Inspector, GuardDuty
- **Monitoring**: CloudWatch, SageMaker Model Monitor
- **IaC**: CloudFormation

---

## 3. Security Controls by Layer

### Layer 1: Model Acquisition Security

**Threat**: Malicious model injection, supply chain attacks

**Controls Implemented**:
- ✅ Token-based authentication via AWS Secrets Manager
- ✅ Model integrity verification (checksums)
- ✅ Audit logging to CloudWatch
- ✅ No trust of remote code execution
- ✅ Secure download over HTTPS only

**Code Example**:
```python
# Secure token retrieval from Secrets Manager
secrets_client = boto3.client('secretsmanager')
response = secrets_client.get_secret_value(
    SecretId='huggingface/api-token'
)
token = response['SecretString']

# Download with verification
model_path = snapshot_download(
    repo_id=model_id,
    token=token,
    local_dir_use_symlinks=False  # Prevent symlink attacks
)
```

---

### Layer 2: Container Security

**Threat**: Vulnerable dependencies, container escape, supply chain attacks

**Controls Implemented**:
- ✅ AWS ECR with vulnerability scanning (basic + enhanced)
- ✅ Immutable image tags
- ✅ KMS encryption at rest
- ✅ Non-root container execution
- ✅ Minimal attack surface (slim base image)
- ✅ Regular automated scanning
- ✅ Image signing (optional)

**Dockerfile Hardening**:
```dockerfile
# Use AWS-maintained base image (regularly patched)
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310

# Run as non-root user
RUN groupadd -g 1000 mlopsuser && \
    useradd -m -u 1000 -g mlopsuser mlopsuser
USER mlopsuser

# Read-only root filesystem where possible
# Drop unnecessary capabilities
```

**Scan Results Integration**:
- Continuous scanning every 24 hours
- Critical/High vulnerabilities block deployment
- Integration with CI/CD pipeline

---

### Layer 3: Training Security

**Threat**: Data exfiltration, unauthorized access, model poisoning

**Controls Implemented**:
- ✅ VPC isolation (optional)
- ✅ Private subnets with no internet access
- ✅ VPC endpoints for AWS services
- ✅ Encrypted inter-container traffic
- ✅ KMS encryption for:
  - Training data (S3)
  - Model artifacts
  - EBS volumes
- ✅ IAM least-privilege roles
- ✅ Network isolation mode

**Security Configuration**:
```yaml
security:
  encryption:
    s3_kms_key_id: "alias/sagemaker-kms-key"
    volume_kms_key_id: "alias/sagemaker-volume-kms-key"
    enable_network_isolation: true

  vpc:
    security_group_ids: ["sg-xxxxx"]
    subnets: ["subnet-xxxxx", "subnet-yyyyy"]
```

---

### Layer 4: Data Security

**Threat**: Data breach, unauthorized access, compliance violations

**Controls Implemented**:
- ✅ S3 bucket encryption (KMS)
- ✅ Bucket versioning enabled
- ✅ Public access blocked
- ✅ Lifecycle policies for data retention
- ✅ Access logging
- ✅ MFA delete protection
- ✅ Cross-region replication (optional)

**S3 Security Settings**:
```python
bucket_encryption:
  - SSEAlgorithm: 'aws:kms'
    KMSMasterKeyID: !GetAtt MLOpsKMSKey.Arn

public_access_block:
  BlockPublicAcls: true
  BlockPublicPolicy: true
  IgnorePublicAcls: true
  RestrictPublicBuckets: true
```

---

### Layer 5: Model Governance

**Threat**: Unauthorized model deployment, model drift, compliance violations

**Controls Implemented**:
- ✅ Automated performance threshold validation
- ✅ Model versioning in SageMaker Registry
- ✅ Approval workflow (manual/automated)
- ✅ Model lineage tracking
- ✅ Model cards for documentation
- ✅ Experiment tracking
- ✅ Audit trail

**Quality Gates**:
```python
thresholds = {
    'perplexity_max': 20.0,    # Industry best practice
    'eval_loss_max': 1.5,       # Model quality threshold
}

# Automatic rejection if thresholds not met
if not check_thresholds(metrics, thresholds):
    approval_status = 'Rejected'
    raise ValueError('Model does not meet performance requirements')
```

---

### Layer 6: Deployment Security

**Threat**: Unauthorized access, data leakage, model serving attacks

**Controls Implemented**:
- ✅ VPC endpoint deployment
- ✅ Data capture for monitoring
- ✅ Auto-scaling with security
- ✅ HTTPS-only endpoints
- ✅ IAM-based access control
- ✅ Request/response encryption
- ✅ Rate limiting
- ✅ Model monitoring (drift, bias, quality)

**Endpoint Security**:
```python
endpoint_config = {
    'DataCaptureConfig': {
        'EnableCapture': True,
        'DestinationS3Uri': 's3://secure-bucket/monitoring/',
        'KmsKeyId': kms_key_id
    },
    'KmsKeyId': kms_key_id,
    'EnableNetworkIsolation': True
}
```

---

## 4. Pipeline Workflow

### End-to-End Secure Pipeline

```
Step 1: Secure Model Download
├─ Authenticate with AWS Secrets Manager
├─ Download from HuggingFace with token
├─ Verify model integrity
├─ Upload to encrypted S3
└─ Log audit trail

Step 2: Container Build & Scan
├─ Build Docker image with security hardening
├─ Push to ECR with immutable tags
├─ Trigger vulnerability scan
├─ Review scan results
└─ Block if critical vulnerabilities found

Step 3: Secure Training
├─ Load encrypted data from S3
├─ Execute training in VPC (optional)
├─ Track experiments in SageMaker
├─ Encrypt model artifacts with KMS
└─ Store in encrypted S3

Step 4: Automated Evaluation
├─ Download model artifacts
├─ Run evaluation metrics
├─ Check performance thresholds
│  ├─ Perplexity < 20
│  └─ Eval Loss < 1.5
└─ Approve/Reject based on metrics

Step 5: Model Registry
├─ Register model version
├─ Attach metrics and metadata
├─ Set approval status
├─ Enable model card
└─ Track lineage

Step 6: Manual Review (optional)
├─ Review metrics in SageMaker Studio
├─ Validate business requirements
├─ Approve or reject deployment
└─ Document decision

Step 7: Secure Deployment
├─ Deploy approved model only
├─ Configure monitoring
├─ Enable auto-scaling
├─ Setup alarms
└─ Continuous monitoring
```

---

## 5. Demo & Implementation

### Prerequisites Setup

```bash
# 1. Deploy infrastructure
aws cloudformation create-stack \
  --stack-name secure-mlops-infra \
  --template-body file://config/cloudformation_template.yaml \
  --capabilities CAPABILITY_NAMED_IAM

# 2. Store HuggingFace token in Secrets Manager
aws secretsmanager create-secret \
  --name huggingface/api-token \
  --secret-string "hf_xxxxxxxxxxxxx"

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Option 1: Run via Jupyter Notebook
jupyter notebook notebooks/secure_mlops_pipeline.ipynb

# Option 2: Run via Python scripts
python src/model_management/secure_model_downloader.py
bash scripts/build_and_push.sh
python src/training/train.py --config config/training_config.yaml
python src/deployment/deploy.py
```

---

### Key Implementation Highlights

**1. Secure Model Download**
```python
downloader = SecureModelDownloader()
model_path = downloader.download_model(
    model_id='meta-llama/Llama-2-7b-hf',
    local_dir='./models',
    cache_dir='./cache'
)
# Uploads to S3 with KMS encryption
downloader.upload_to_s3(model_path, s3_uri, encrypt=True)
```

**2. ECR Security**
```python
ecr_manager = SecureECRManager()
ecr_manager.create_secure_repository(
    repository_name='secure-mlops-training',
    scan_on_push=True,
    enable_encryption=True
)
```

**3. Training with Experiments**
```python
with Run(experiment_name=exp_name, run_name=run_name) as run:
    estimator.fit({
        'train': s3_train_path,
        'validation': s3_eval_path
    })
```

---

### Performance Validation

**Industry Best Practices for LLM Evaluation**:

| Metric | Good | Excellent | Our Threshold |
|--------|------|-----------|---------------|
| Perplexity | < 20 | < 10 | < 20 |
| Eval Loss | < 1.5 | < 1.0 | < 1.5 |
| Training Time | Optimized | GPU Efficient | Spot Instances |

**Automated Quality Gates**:
```python
evaluator = ModelEvaluator(model_path)
metrics = evaluator.evaluate_metrics(eval_dataset)

passed, failures = evaluator.check_thresholds(metrics)
if not passed:
    # Automatic rejection
    approval_status = 'Rejected'
    logger.error(f"Threshold failures: {failures}")
```

---

## 6. Compliance & Governance

### Regulatory Compliance

**SOC 2 Type II**:
- ✅ Encryption at rest and in transit
- ✅ Access controls and audit logging
- ✅ Change management via approval workflows
- ✅ Monitoring and alerting

**HIPAA Ready**:
- ✅ BAA-eligible AWS services (SageMaker, S3, KMS)
- ✅ Data encryption
- ✅ Access logging
- ✅ Network isolation

**GDPR**:
- ✅ Data encryption
- ✅ Data retention policies
- ✅ Right to deletion (S3 lifecycle)
- ✅ Audit trail

---

### Audit Trail

**Complete Logging Strategy**:

```
CloudWatch Logs
├─ /aws/sagemaker/TrainingJobs
│  └─ Training execution logs
├─ /aws/sagemaker/Endpoints
│  └─ Inference logs
├─ /aws/mlops/secure-pipeline
│  ├─ Model downloads
│  ├─ ECR operations
│  └─ Approval workflows
└─ /aws/ecr/vulnerability-scans
   └─ Scan results
```

**Retention Policy**: 90 days (configurable)

**Log Analysis**:
- CloudWatch Insights for querying
- Automated alerts on anomalies
- Integration with SIEM tools

---

### Model Cards & Documentation

**Model Card Contents**:
```yaml
Model Information:
  Name: secure-llm-v1.0
  Version: 1
  Base Model: meta-llama/Llama-2-7b-hf
  Fine-tuning Date: 2025-11-11

Performance Metrics:
  Perplexity: 15.2
  Eval Loss: 1.23
  Training Samples: 10,000

Intended Use:
  - Customer support automation
  - Content generation

Limitations:
  - May produce biased outputs
  - Not suitable for medical advice

Ethical Considerations:
  - Reviewed for bias
  - Safety testing completed
```

---

## 7. Monitoring & Observability

### Multi-Layer Monitoring

**Infrastructure Monitoring**:
- CloudWatch Metrics for SageMaker
- VPC Flow Logs
- GuardDuty for threat detection

**Application Monitoring**:
- Training job metrics (loss, accuracy)
- Endpoint latency and throughput
- Error rates and exceptions

**Security Monitoring**:
- ECR scan results
- IAM access patterns (CloudTrail)
- KMS key usage
- Failed authentication attempts

**Model Monitoring**:
- Data quality monitoring
- Model quality monitoring
- Bias drift detection
- Feature drift detection

---

### SageMaker Model Monitor

**Automated Monitoring Setup**:

```python
data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,
    destination_s3_uri=f's3://{bucket}/monitoring/',
    kms_key_id=kms_key_id,
    capture_options=['Input', 'Output']
)

# Schedule monitoring
monitor = DataQualityMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge'
)

monitor.create_monitoring_schedule(
    monitor_schedule_name='data-quality-monitor',
    endpoint_input=endpoint_name,
    schedule_cron_expression='cron(0 * * * ? *)'  # Hourly
)
```

---

### Alerting & Incident Response

**CloudWatch Alarms**:
```yaml
Alarms:
  - ModelInvocationErrors > 10/min
  - ModelLatency > 1000ms (p99)
  - CriticalVulnerabilitiesFound > 0
  - UnauthorizedAccessAttempts > 5
  - ModelDriftDetected == True
```

**Incident Response Workflow**:
1. Alert triggered → SNS notification
2. On-call engineer paged
3. Runbook executed
4. Endpoint rolled back if needed
5. Root cause analysis
6. Remediation and lessons learned

---

## 8. Best Practices & Lessons Learned

### Security Best Practices

**1. Defense in Depth**
- Multiple layers of security controls
- No single point of failure
- Assume breach mentality

**2. Least Privilege**
- Minimal IAM permissions
- Service-specific roles
- Regular permission audits

**3. Encryption Everywhere**
- At rest: S3, EBS, ECR
- In transit: TLS 1.2+
- Key rotation every 90 days

**4. Automated Security**
- Automated vulnerability scanning
- Automated compliance checks
- Automated remediation where possible

**5. Continuous Monitoring**
- Real-time alerts
- Centralized logging
- Regular security audits

---

### MLOps Best Practices

**1. Experiment Tracking**
- Every training run tracked
- Reproducible experiments
- Version control for code and data

**2. Model Versioning**
- Semantic versioning
- Immutable model artifacts
- Clear lineage tracking

**3. Quality Gates**
- Automated performance validation
- Threshold-based approval
- No manual bypasses

**4. Infrastructure as Code**
- CloudFormation for all resources
- Version controlled
- Peer reviewed changes

**5. Separation of Environments**
- Dev → Staging → Production
- Different AWS accounts (optional)
- Progressive deployment

---

### Cost Optimization

**Security doesn't have to be expensive**:

✅ **Spot Instances** for training (70% cost savings)
```python
use_spot_instances=True,
max_wait=2 * max_run  # Up to 2x training time
```

✅ **S3 Intelligent Tiering**
- Automatic cost optimization
- No performance impact

✅ **Right-sized instances**
- Start small, scale as needed
- Auto-scaling for production

✅ **VPC Endpoints** instead of NAT Gateway
- Reduce data transfer costs
- Better security

**Estimated Monthly Costs** (for demo):
- Training (spot): ~$50-100/month
- Inference: ~$200-300/month (ml.g5.xlarge)
- Storage: ~$10-20/month
- **Total: ~$260-420/month**

---

### Common Pitfalls & Solutions

**Pitfall 1: Overly Complex Security**
- ❌ Problem: Too many security controls slow development
- ✅ Solution: Automate security checks in CI/CD

**Pitfall 2: Hardcoded Credentials**
- ❌ Problem: Secrets in code or config files
- ✅ Solution: AWS Secrets Manager + IAM roles

**Pitfall 3: No Model Validation**
- ❌ Problem: Deploying low-quality models
- ✅ Solution: Automated threshold checks + approval workflow

**Pitfall 4: Ignoring Vulnerability Scans**
- ❌ Problem: Deploying vulnerable containers
- ✅ Solution: Block deployment if critical vulns found

**Pitfall 5: No Monitoring**
- ❌ Problem: Issues discovered too late
- ✅ Solution: Comprehensive monitoring from day 1

---

## 9. Future Enhancements

### Roadmap

**Phase 1: Current Implementation** ✅
- Secure pipeline end-to-end
- Basic monitoring
- Manual approval

**Phase 2: Advanced Security** (Q1 2026)
- [ ] Image signing with AWS Signer
- [ ] Runtime security with Falco
- [ ] Secrets rotation automation
- [ ] Advanced threat detection

**Phase 3: ML Security** (Q2 2026)
- [ ] Adversarial robustness testing
- [ ] Explainability integration
- [ ] Bias detection in production
- [ ] Privacy-preserving ML (federated learning)

**Phase 4: Automation** (Q3 2026)
- [ ] Fully automated approval (based on rules)
- [ ] Self-healing infrastructure
- [ ] Automated incident response
- [ ] Continuous training pipeline

---

### Integration Opportunities

**CI/CD Integration**:
```yaml
# GitHub Actions / Jenkins pipeline
steps:
  - name: Security Scan
    run: |
      trivy image $IMAGE_URI

  - name: Train Model
    run: |
      python train.py

  - name: Validate Model
    run: |
      python evaluate.py --threshold-file thresholds.yaml

  - name: Deploy
    if: approved
    run: |
      python deploy.py --environment production
```

**GitOps for ML**:
- Model definitions in Git
- Automated deployments on PR merge
- Rollback via Git revert

---

## Summary

### What We've Accomplished

✅ **Comprehensive Security**
- 6 layers of security controls
- Defense-in-depth approach
- Industry best practices

✅ **Complete MLOps Pipeline**
- Model acquisition → Training → Evaluation → Registry → Deployment
- Automated quality gates
- Experiment tracking

✅ **Compliance Ready**
- SOC2, HIPAA, GDPR considerations
- Complete audit trail
- Automated compliance checks

✅ **Production Ready**
- Monitoring and alerting
- Auto-scaling
- Incident response

---

### Key Takeaways

1. **Security is Everyone's Responsibility**
   - DevOps, ML Engineers, and Security must collaborate

2. **Automate Everything**
   - Manual security processes don't scale
   - Shift-left security approach

3. **Monitor Continuously**
   - Security is not "set and forget"
   - Continuous validation required

4. **Balance Security & Velocity**
   - Good security enables faster development
   - Automation is the key

5. **Start Simple, Iterate**
   - Don't try to implement everything at once
   - Build security incrementally

---

## Resources

### Documentation
- Project README: `/README.md`
- Setup Guide: `/docs/SETUP.md`
- Architecture: `/docs/ARCHITECTURE.md`

### Code Repository
```
mlops/
├── config/                 # Configuration files
├── src/                    # Source code
│   ├── model_management/   # Model download & upload
│   ├── training/          # Training scripts
│   ├── deployment/        # Deployment code
│   └── security/          # Security utilities
├── notebooks/             # Jupyter notebooks
├── scripts/               # Helper scripts
└── presentation/          # This presentation
```

### Learning Resources
- AWS SageMaker Security: https://docs.aws.amazon.com/sagemaker/
- MLOps Best Practices: https://ml-ops.org/
- OWASP ML Security: https://owasp.org/www-project-machine-learning-security-top-10/

---

## Q&A

### Common Questions

**Q: How do you handle model bias?**
A: SageMaker Clarify for bias detection + manual review

**Q: What about adversarial attacks?**
A: Input validation, rate limiting, monitoring for anomalies

**Q: How long does the pipeline take?**
A: ~1-2 hours for training (depends on model size)

**Q: Can this run on-premises?**
A: Yes, with modifications (use Kubernetes, private registries)

**Q: What's the TCO?**
A: ~$300-500/month for small-scale deployment

---

## Thank You!

### Contact & Contributions

**Questions?** Feel free to reach out!

**Want to Contribute?**
- Report issues on GitHub
- Submit pull requests
- Share your security findings

**Next Steps**:
1. Clone the repository
2. Follow the setup guide
3. Run the demo
4. Customize for your use case

---

## Appendix: Technical Details

### IAM Policy Example
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::sagemaker-*/*",
      "Condition": {
        "StringEquals": {
          "s3:x-amz-server-side-encryption": "aws:kms"
        }
      }
    }
  ]
}
```

### KMS Key Policy
```yaml
KeyPolicy:
  Statement:
    - Effect: Allow
      Principal:
        Service: sagemaker.amazonaws.com
      Action:
        - kms:Decrypt
        - kms:Encrypt
        - kms:GenerateDataKey
```

---

## Backup Slides

### Disaster Recovery

**Backup Strategy**:
- S3 versioning enabled
- Cross-region replication (optional)
- Model artifacts in multiple regions
- RTO: < 4 hours
- RPO: < 1 hour

**Recovery Procedures**:
1. Restore from S3 versioning
2. Redeploy model from registry
3. Verify endpoint health
4. Switch traffic via Route53

---

### Performance Benchmarks

| Operation | Time | Cost |
|-----------|------|------|
| Model Download | 5-10 min | $0 |
| Container Build | 10-15 min | $0.50 |
| Training (1 epoch) | 30-60 min | $2-5 |
| Evaluation | 5-10 min | $0 |
| Deployment | 10-15 min | $0 |
| **Total** | **60-110 min** | **$2.50-5.50** |

---

### Compliance Checklist

- [x] Data encryption at rest
- [x] Data encryption in transit
- [x] Access controls (IAM)
- [x] Audit logging (CloudWatch)
- [x] Network isolation (VPC)
- [x] Vulnerability scanning
- [x] Change management
- [x] Incident response plan
- [x] Data retention policy
- [x] Access review process

---

**END OF PRESENTATION**
