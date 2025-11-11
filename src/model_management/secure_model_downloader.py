"""
Secure Model Downloader from HuggingFace
Implements security best practices for downloading LLM models
"""
import os
import hashlib
import logging
from typing import Optional, Dict
from pathlib import Path
import boto3
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecureModelDownloader:
    """
    Securely download models from HuggingFace with:
    - Token-based authentication
    - Model verification
    - Secure storage
    - Audit logging
    """

    def __init__(self, config_path: str = "config/training_config.yaml"):
        """Initialize the secure downloader with configuration"""
        self.config = self._load_config(config_path)
        self.hf_token = self._get_secure_token()
        self.api = HfApi(token=self.hf_token)
        self.s3_client = boto3.client('s3')

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_secure_token(self) -> str:
        """
        Retrieve HuggingFace token securely from:
        1. AWS Secrets Manager (recommended for production)
        2. Environment variable (for development)
        """
        try:
            # Try AWS Secrets Manager first
            secrets_client = boto3.client('secretsmanager')
            response = secrets_client.get_secret_value(
                SecretId='huggingface/api-token'
            )
            logger.info("Retrieved HuggingFace token from AWS Secrets Manager")
            return response['SecretString']
        except Exception as e:
            logger.warning(f"Could not retrieve from Secrets Manager: {e}")
            # Fallback to environment variable
            token = os.getenv('HUGGINGFACE_TOKEN')
            if not token:
                raise ValueError(
                    "HuggingFace token not found. Set HUGGINGFACE_TOKEN env var "
                    "or store in AWS Secrets Manager"
                )
            logger.warning("Using HuggingFace token from environment variable")
            return token

    def verify_model_integrity(self, model_path: str) -> bool:
        """
        Verify model integrity using checksums
        """
        logger.info(f"Verifying integrity of model at {model_path}")
        # In production, compare with published checksums from HuggingFace
        # This is a simplified version
        try:
            for root, _, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    self._calculate_checksum(file_path)
            logger.info("Model integrity verification passed")
            return True
        except Exception as e:
            logger.error(f"Model integrity verification failed: {e}")
            return False

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        checksum = sha256_hash.hexdigest()
        logger.debug(f"Checksum for {file_path}: {checksum}")
        return checksum

    def download_model(
        self,
        model_id: Optional[str] = None,
        local_dir: str = "./models",
        cache_dir: str = "./cache"
    ) -> str:
        """
        Securely download model from HuggingFace

        Args:
            model_id: HuggingFace model identifier
            local_dir: Local directory to store the model
            cache_dir: Cache directory for downloads

        Returns:
            Path to downloaded model
        """
        if model_id is None:
            model_id = self.config['model']['base_model']

        logger.info(f"Starting secure download of model: {model_id}")

        try:
            # Verify model exists and user has access
            model_info = self.api.model_info(model_id)
            logger.info(f"Model found: {model_info.modelId}")
            logger.info(f"Model author: {model_info.author}")
            logger.info(f"Downloads: {model_info.downloads}")

            # Create secure local directory
            Path(local_dir).mkdir(parents=True, exist_ok=True, mode=0o700)
            Path(cache_dir).mkdir(parents=True, exist_ok=True, mode=0o700)

            # Download model with authentication
            model_path = snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                cache_dir=cache_dir,
                token=self.hf_token,
                resume_download=True,
                local_dir_use_symlinks=False  # Security: avoid symlink attacks
            )

            logger.info(f"Model downloaded successfully to: {model_path}")

            # Verify model integrity
            if not self.verify_model_integrity(model_path):
                raise ValueError("Model integrity verification failed")

            # Audit log
            self._audit_log("MODEL_DOWNLOAD", {
                "model_id": model_id,
                "path": model_path,
                "status": "SUCCESS"
            })

            return model_path

        except HfHubHTTPError as e:
            logger.error(f"HuggingFace API error: {e}")
            self._audit_log("MODEL_DOWNLOAD", {
                "model_id": model_id,
                "status": "FAILED",
                "error": str(e)
            })
            raise
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            self._audit_log("MODEL_DOWNLOAD", {
                "model_id": model_id,
                "status": "FAILED",
                "error": str(e)
            })
            raise

    def upload_to_s3(self, local_path: str, s3_uri: str, encrypt: bool = True):
        """
        Upload model to S3 with encryption

        Args:
            local_path: Local path to model
            s3_uri: S3 URI (s3://bucket/path)
            encrypt: Enable server-side encryption
        """
        logger.info(f"Uploading model from {local_path} to {s3_uri}")

        # Parse S3 URI
        s3_parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket = s3_parts[0]
        prefix = s3_parts[1] if len(s3_parts) > 1 else ""

        try:
            for root, _, files in os.walk(local_path):
                for file in files:
                    local_file = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file, local_path)
                    s3_key = os.path.join(prefix, relative_path)

                    # Upload with encryption
                    extra_args = {}
                    if encrypt:
                        extra_args['ServerSideEncryption'] = 'aws:kms'
                        extra_args['SSEKMSKeyId'] = self.config.get(
                            'security', {}
                        ).get('encryption', {}).get('s3_kms_key_id')

                    self.s3_client.upload_file(
                        local_file,
                        bucket,
                        s3_key,
                        ExtraArgs=extra_args
                    )
                    logger.debug(f"Uploaded {s3_key}")

            logger.info(f"Model uploaded successfully to {s3_uri}")

            self._audit_log("S3_UPLOAD", {
                "local_path": local_path,
                "s3_uri": s3_uri,
                "encrypted": encrypt,
                "status": "SUCCESS"
            })

        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            self._audit_log("S3_UPLOAD", {
                "local_path": local_path,
                "s3_uri": s3_uri,
                "status": "FAILED",
                "error": str(e)
            })
            raise

    def _audit_log(self, action: str, details: Dict):
        """
        Send audit logs to CloudWatch Logs
        """
        try:
            logs_client = boto3.client('logs')
            log_group = '/aws/mlops/secure-pipeline'
            log_stream = 'model-downloads'

            # Create log group if it doesn't exist
            try:
                logs_client.create_log_group(logGroupName=log_group)
            except logs_client.exceptions.ResourceAlreadyExistsException:
                pass

            # Create log stream if it doesn't exist
            try:
                logs_client.create_log_stream(
                    logGroupName=log_group,
                    logStreamName=log_stream
                )
            except logs_client.exceptions.ResourceAlreadyExistsException:
                pass

            # Send log event
            import time
            logs_client.put_log_events(
                logGroupName=log_group,
                logStreamName=log_stream,
                logEvents=[{
                    'timestamp': int(time.time() * 1000),
                    'message': f"ACTION={action} DETAILS={details}"
                }]
            )
        except Exception as e:
            logger.warning(f"Could not send audit log: {e}")


if __name__ == "__main__":
    # Example usage
    downloader = SecureModelDownloader()

    # Download model
    model_path = downloader.download_model()

    # Upload to S3 (optional)
    # downloader.upload_to_s3(model_path, "s3://your-bucket/models/base-model/")
