# Secure MLOps Pipeline for LLM Fine-Tuning

A production-grade, security-focused MLOps pipeline demonstrating industry best practices for secure machine learning operations on AWS SageMaker.

![Security](https://img.shields.io/badge/security-hardened-success)
![AWS](https://img.shields.io/badge/AWS-SageMaker-orange)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This project implements a **complete secure MLOps pipeline** with comprehensive security controls at every stage, from model acquisition to production deployment. Built on AWS SageMaker, it demonstrates security engineering and security operations best practices for machine learning workloads.

### Key Features

- **🔒 Security-First Design**: Defense-in-depth with 6 layers of security controls
- **🚀 Production-Ready**: Complete CI/CD pipeline with automated quality gates
- **📊 Full Observability**: Comprehensive monitoring, logging, and audit trail
- **✅ Compliance**: SOC2, HIPAA, and GDPR considerations built-in
- **🔄 Automated Workflows**: From model download to deployment with minimal manual intervention
- **📈 Experiment Tracking**: Complete lineage and versioning in SageMaker Experiments
- **🎯 Quality Gates**: Automated performance threshold validation based on industry standards

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SECURE MLOPS PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. Model Source        →  HuggingFace (Secure Token Auth)      │
│                            ↓                                      │
│  2. Container Registry  →  AWS ECR (Vuln Scanning + Encryption) │
│                            ↓                                      │
│  3. Training           →  SageMaker (VPC + KMS + IAM)           │
│                            ↓                                      │
│  4. Evaluation         →  Automated Threshold Validation         │
│                            ↓                                      │
│  5. Model Registry     →  SageMaker Registry (Approval Flow)    │
│                            ↓                                      │
│  6. Deployment         →  Secure Endpoint (Monitoring)          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Security Controls

### Layer 1: Model Acquisition
- ✅ Token-based authentication via AWS Secrets Manager
- ✅ Model integrity verification with checksums
- ✅ Audit logging to CloudWatch
- ✅ No remote code execution
- ✅ Encrypted storage in S3

### Layer 2: Container Security
- ✅ ECR vulnerability scanning (basic + enhanced with Inspector)
- ✅ Immutable image tags
- ✅ KMS encryption at rest
- ✅ Non-root container execution
- ✅ Minimal attack surface
- ✅ Regular automated scanning

### Layer 3: Training Security
- ✅ VPC isolation with private subnets
- ✅ VPC endpoints for AWS services (no internet access)
- ✅ Encrypted inter-container traffic
- ✅ KMS encryption for volumes and model artifacts
- ✅ IAM least-privilege roles
- ✅ Network isolation mode

### Layer 4: Data Security
- ✅ S3 bucket encryption with KMS
- ✅ Versioning enabled
- ✅ Public access blocked
- ✅ Access logging
- ✅ Lifecycle policies

### Layer 5: Model Governance
- ✅ Automated performance threshold validation
- ✅ Model versioning in SageMaker Registry
- ✅ Approval workflows
- ✅ Model lineage tracking
- ✅ Experiment tracking
- ✅ Complete audit trail

### Layer 6: Deployment Security
- ✅ VPC endpoint deployment
- ✅ Data capture for monitoring
- ✅ Auto-scaling with security
- ✅ HTTPS-only endpoints
- ✅ IAM-based access control
- ✅ Model monitoring (drift, bias, quality)

## Project Structure

```
mlops/
├── config/                     # Configuration files
│   ├── security_config.yaml    # Security settings
│   ├── training_config.yaml    # Training hyperparameters
│   ├── iam_policies.json       # IAM policy definitions
│   └── cloudformation_template.yaml  # Infrastructure as Code
│
├── src/                        # Source code
│   ├── model_management/
│   │   └── secure_model_downloader.py  # Secure HuggingFace downloads
│   ├── training/
│   │   ├── train.py            # Training script with security
│   │   └── evaluator.py        # Model evaluation
│   ├── deployment/
│   │   ├── model_registry.py   # Model versioning & registry
│   │   └── deploy.py           # Secure endpoint deployment
│   └── security/
│       └── ecr_manager.py      # ECR security management
│
├── scripts/
│   └── build_and_push.sh       # Container build & push script
│
├── notebooks/
│   └── secure_mlops_pipeline.ipynb  # End-to-end demo notebook
│
├── presentation/
│   └── SECURE_MLOPS_PRESENTATION.md  # Complete presentation
│
├── docs/
│   └── SETUP.md                # Setup instructions
│
├── Dockerfile                  # Hardened container image
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Quick Start

### Prerequisites

- AWS Account with appropriate permissions
- AWS CLI configured
- Python 3.10+
- Docker
- HuggingFace account (for model access)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd mlops
```

2. **Deploy AWS infrastructure**
```bash
aws cloudformation create-stack \
  --stack-name secure-mlops-infra \
  --template-body file://config/cloudformation_template.yaml \
  --capabilities CAPABILITY_NAMED_IAM
```

3. **Store HuggingFace token**
```bash
aws secretsmanager create-secret \
  --name huggingface/api-token \
  --secret-string "your_huggingface_token"
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Run the pipeline**
```bash
jupyter notebook notebooks/secure_mlops_pipeline.ipynb
```

For detailed setup instructions, see [docs/SETUP.md](docs/SETUP.md)

## Usage

### Option 1: Jupyter Notebook (Recommended)

The easiest way to run the entire pipeline is through the provided Jupyter notebook:

```bash
jupyter notebook notebooks/secure_mlops_pipeline.ipynb
```

This notebook provides:
- Step-by-step execution
- Inline documentation
- Security best practices
- Complete end-to-end workflow

### Option 2: Python Scripts

Run individual components:

```bash
# 1. Download model securely
python -m src.model_management.secure_model_downloader

# 2. Build and push container
./scripts/build_and_push.sh secure-mlops-training v1.0

# 3. Run training (see notebook for complete example)

# 4. Deploy endpoint (see notebook for complete example)
```

## Performance Thresholds

The pipeline validates models against industry best practices:

| Metric | Threshold | Industry Benchmark |
|--------|-----------|-------------------|
| Perplexity | < 20.0 | Good: <20, Excellent: <10 |
| Eval Loss | < 1.5 | Good: <1.5, Excellent: <1.0 |

Models that don't meet thresholds are automatically rejected and must be retrained.

## Monitoring & Observability

### Logging
- **CloudWatch Logs**: Training jobs, endpoints, security events
- **Audit Trail**: Complete lineage from download to deployment
- **Retention**: 90 days (configurable)

### Metrics
- **Training Metrics**: Loss, accuracy, perplexity
- **Endpoint Metrics**: Latency, throughput, errors
- **Security Metrics**: Vulnerability scans, access attempts

### Alerts
- Model performance degradation
- Security vulnerabilities detected
- Unauthorized access attempts
- Resource limit exceeded

## Compliance

This pipeline implements controls for:

- **SOC 2 Type II**: Encryption, access controls, audit logging
- **HIPAA**: BAA-eligible services, PHI protection
- **GDPR**: Data encryption, retention policies, audit trail
- **PCI-DSS**: Network isolation, encryption, access control

## Cost Optimization

Estimated monthly costs (AWS us-east-1):

- **Development**: ~$50-100/month
- **Production**: ~$300-500/month

Cost-saving features:
- Spot instances for training (70% savings)
- S3 Intelligent Tiering
- Auto-scaling for endpoints
- VPC endpoints (vs NAT Gateway)

## Documentation

- **[Setup Guide](docs/SETUP.md)**: Detailed installation instructions
- **[Presentation](presentation/SECURE_MLOPS_PRESENTATION.md)**: Complete technical presentation
- **[Notebook](notebooks/secure_mlops_pipeline.ipynb)**: Interactive demo

## Security Best Practices Implemented

1. **Defense in Depth**: Multiple layers of security controls
2. **Least Privilege**: Minimal IAM permissions
3. **Encryption Everywhere**: At rest and in transit
4. **Automated Security**: Vulnerability scanning, compliance checks
5. **Continuous Monitoring**: Real-time alerts and logging
6. **Immutable Infrastructure**: Infrastructure as Code
7. **Secrets Management**: AWS Secrets Manager integration
8. **Network Isolation**: VPC with private subnets
9. **Audit Logging**: Complete trail in CloudWatch
10. **Quality Gates**: Automated performance validation

## Troubleshooting

See [docs/SETUP.md#troubleshooting](docs/SETUP.md#troubleshooting) for common issues and solutions.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Implement security best practices
4. Add tests where applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- AWS SageMaker team for excellent documentation
- HuggingFace for model hosting
- MLOps community for best practices

## Support

- **Issues**: GitHub Issues
- **Documentation**: See `/docs` directory
- **AWS Support**: https://console.aws.amazon.com/support/

---

**Built with ❤️ and 🔒 by the MLOps Security Team**
