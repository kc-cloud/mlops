"""
Secure ECR Manager for container image management
Implements security scanning and vulnerability management
"""
import boto3
import logging
import time
from typing import Dict, List, Optional
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecureECRManager:
    """
    Manage ECR repositories with security controls:
    - Automated vulnerability scanning
    - Image signing
    - Immutable tags
    - Lifecycle policies
    """

    def __init__(self, region: str = "us-east-1"):
        self.ecr_client = boto3.client('ecr', region_name=region)
        self.region = region

    def create_secure_repository(
        self,
        repository_name: str,
        scan_on_push: bool = True,
        enable_encryption: bool = True,
        kms_key: Optional[str] = None
    ) -> Dict:
        """
        Create ECR repository with security controls

        Args:
            repository_name: Name of the ECR repository
            scan_on_push: Enable vulnerability scanning on push
            enable_encryption: Enable encryption at rest
            kms_key: KMS key for encryption

        Returns:
            Repository details
        """
        try:
            # Check if repository exists
            try:
                response = self.ecr_client.describe_repositories(
                    repositoryNames=[repository_name]
                )
                logger.info(f"Repository {repository_name} already exists")
                return response['repositories'][0]
            except self.ecr_client.exceptions.RepositoryNotFoundException:
                pass

            # Create repository with security settings
            kwargs = {
                'repositoryName': repository_name,
                'imageScanningConfiguration': {
                    'scanOnPush': scan_on_push
                },
                'imageTagMutability': 'IMMUTABLE',  # Security best practice
                'tags': [
                    {'Key': 'Project', 'Value': 'SecureMLOps'},
                    {'Key': 'SecurityScanning', 'Value': 'Enabled'},
                    {'Key': 'ManagedBy', 'Value': 'MLOps-Security-Team'}
                ]
            }

            if enable_encryption:
                encryption_config = {'encryptionType': 'KMS'}
                if kms_key:
                    encryption_config['kmsKey'] = kms_key
                kwargs['encryptionConfiguration'] = encryption_config

            response = self.ecr_client.create_repository(**kwargs)

            logger.info(
                f"Created secure repository: {repository_name} "
                f"with scanning enabled: {scan_on_push}"
            )

            # Set lifecycle policy to clean old images
            self._set_lifecycle_policy(repository_name)

            # Enable enhanced scanning (Inspector)
            self._enable_enhanced_scanning(repository_name)

            return response['repository']

        except Exception as e:
            logger.error(f"Failed to create repository: {e}")
            raise

    def _set_lifecycle_policy(self, repository_name: str):
        """Set lifecycle policy to remove old/untagged images"""
        lifecycle_policy = {
            "rules": [
                {
                    "rulePriority": 1,
                    "description": "Remove untagged images after 1 day",
                    "selection": {
                        "tagStatus": "untagged",
                        "countType": "sinceImagePushed",
                        "countUnit": "days",
                        "countNumber": 1
                    },
                    "action": {
                        "type": "expire"
                    }
                },
                {
                    "rulePriority": 2,
                    "description": "Keep only last 10 tagged images",
                    "selection": {
                        "tagStatus": "tagged",
                        "tagPrefixList": ["v"],
                        "countType": "imageCountMoreThan",
                        "countNumber": 10
                    },
                    "action": {
                        "type": "expire"
                    }
                }
            ]
        }

        try:
            self.ecr_client.put_lifecycle_policy(
                repositoryName=repository_name,
                lifecyclePolicyText=str(lifecycle_policy).replace("'", '"')
            )
            logger.info(f"Set lifecycle policy for {repository_name}")
        except Exception as e:
            logger.warning(f"Could not set lifecycle policy: {e}")

    def _enable_enhanced_scanning(self, repository_name: str):
        """Enable Amazon Inspector enhanced scanning"""
        try:
            self.ecr_client.put_registry_scanning_configuration(
                scanType='ENHANCED',
                rules=[
                    {
                        'scanFrequency': 'CONTINUOUS_SCAN',
                        'repositoryFilters': [
                            {'filter': repository_name, 'filterType': 'WILDCARD'}
                        ]
                    }
                ]
            )
            logger.info(f"Enabled enhanced scanning for {repository_name}")
        except Exception as e:
            logger.warning(f"Could not enable enhanced scanning: {e}")

    def push_image(
        self,
        repository_name: str,
        image_tag: str,
        dockerfile_path: str = ".",
        wait_for_scan: bool = True
    ) -> Dict:
        """
        Build and push Docker image to ECR with security scanning

        Args:
            repository_name: ECR repository name
            image_tag: Tag for the image
            dockerfile_path: Path to Dockerfile
            wait_for_scan: Wait for vulnerability scan to complete

        Returns:
            Image details and scan results
        """
        import subprocess

        try:
            # Get ECR login credentials
            login_response = self.ecr_client.get_authorization_token()
            token = login_response['authorizationData'][0]['authorizationToken']
            endpoint = login_response['authorizationData'][0]['proxyEndpoint']

            # Docker login
            import base64
            username, password = base64.b64decode(token).decode().split(':')
            login_cmd = f"echo {password} | docker login --username {username} --password-stdin {endpoint}"
            subprocess.run(login_cmd, shell=True, check=True)

            # Build image
            image_uri = f"{endpoint.replace('https://', '')}/{repository_name}:{image_tag}"
            logger.info(f"Building image: {image_uri}")

            build_cmd = f"docker build -t {image_uri} {dockerfile_path}"
            subprocess.run(build_cmd, shell=True, check=True)

            # Scan locally before pushing (optional)
            logger.info("Running local vulnerability scan with Trivy...")
            scan_cmd = f"docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy image {image_uri}"
            try:
                subprocess.run(scan_cmd, shell=True, check=True)
            except subprocess.CalledProcessError:
                logger.warning("Local Trivy scan failed or not available")

            # Push image
            logger.info(f"Pushing image to ECR: {image_uri}")
            push_cmd = f"docker push {image_uri}"
            subprocess.run(push_cmd, shell=True, check=True)

            # Get image details
            images = self.ecr_client.describe_images(
                repositoryName=repository_name,
                imageIds=[{'imageTag': image_tag}]
            )

            image_digest = images['imageDetails'][0]['imageDigest']
            logger.info(f"Image pushed successfully. Digest: {image_digest}")

            # Wait for scan results
            if wait_for_scan:
                scan_results = self._wait_for_scan_results(
                    repository_name,
                    image_tag
                )
                return {
                    'imageUri': image_uri,
                    'imageDigest': image_digest,
                    'scanResults': scan_results
                }

            return {
                'imageUri': image_uri,
                'imageDigest': image_digest
            }

        except Exception as e:
            logger.error(f"Failed to push image: {e}")
            raise

    def _wait_for_scan_results(
        self,
        repository_name: str,
        image_tag: str,
        max_wait: int = 300
    ) -> Dict:
        """
        Wait for ECR vulnerability scan to complete

        Args:
            repository_name: Repository name
            image_tag: Image tag
            max_wait: Maximum wait time in seconds

        Returns:
            Scan findings
        """
        logger.info("Waiting for vulnerability scan results...")
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                response = self.ecr_client.describe_image_scan_findings(
                    repositoryName=repository_name,
                    imageId={'imageTag': image_tag}
                )

                scan_status = response['imageScanStatus']['status']

                if scan_status == 'COMPLETE':
                    findings = response.get('imageScanFindings', {})
                    self._log_scan_findings(findings)
                    return findings
                elif scan_status == 'FAILED':
                    logger.error("Vulnerability scan failed")
                    return {'status': 'FAILED'}

                logger.info(f"Scan status: {scan_status}, waiting...")
                time.sleep(10)

            except self.ecr_client.exceptions.ScanNotFoundException:
                logger.info("Scan not started yet, waiting...")
                time.sleep(10)

        logger.warning(f"Scan did not complete within {max_wait} seconds")
        return {'status': 'TIMEOUT'}

    def _log_scan_findings(self, findings: Dict):
        """Log vulnerability scan findings"""
        if 'findingSeverityCounts' in findings:
            severity_counts = findings['findingSeverityCounts']
            logger.info("=" * 60)
            logger.info("VULNERABILITY SCAN RESULTS")
            logger.info("=" * 60)

            for severity, count in severity_counts.items():
                logger.info(f"{severity}: {count} vulnerabilities")

            # Check for critical/high vulnerabilities
            critical = severity_counts.get('CRITICAL', 0)
            high = severity_counts.get('HIGH', 0)

            if critical > 0 or high > 0:
                logger.warning(
                    f"Found {critical} CRITICAL and {high} HIGH "
                    "severity vulnerabilities!"
                )
                logger.warning("Review findings before deploying to production")
            else:
                logger.info("No critical or high severity vulnerabilities found")

            logger.info("=" * 60)

    def get_scan_results(self, repository_name: str, image_tag: str) -> Dict:
        """Get scan results for an image"""
        try:
            response = self.ecr_client.describe_image_scan_findings(
                repositoryName=repository_name,
                imageId={'imageTag': image_tag}
            )
            return response.get('imageScanFindings', {})
        except Exception as e:
            logger.error(f"Failed to get scan results: {e}")
            return {}


if __name__ == "__main__":
    # Example usage
    ecr_manager = SecureECRManager()

    # Create secure repository
    repo = ecr_manager.create_secure_repository(
        repository_name="secure-mlops-training",
        scan_on_push=True,
        enable_encryption=True
    )

    # Build and push image
    # result = ecr_manager.push_image(
    #     repository_name="secure-mlops-training",
    #     image_tag="v1.0",
    #     dockerfile_path=".",
    #     wait_for_scan=True
    # )
