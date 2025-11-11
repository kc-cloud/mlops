"""
SageMaker Model Registry Manager with Versioning and Approval Workflow
"""
import logging
from typing import Dict, Optional, List
import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.model_metrics import ModelMetrics, MetricsSource
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecureModelRegistry:
    """
    Manage models in SageMaker Model Registry with:
    - Versioning
    - Approval workflow
    - Model cards
    - Lineage tracking
    """

    def __init__(self, region: str = "us-east-1"):
        self.sm_client = boto3.client('sagemaker', region_name=region)
        self.session = sagemaker.Session()
        self.region = region

    def create_model_package_group(
        self,
        group_name: str,
        description: str = "Secure MLOps Model Group"
    ) -> str:
        """Create a model package group for versioning"""
        try:
            response = self.sm_client.create_model_package_group(
                ModelPackageGroupName=group_name,
                ModelPackageGroupDescription=description,
                Tags=[
                    {'Key': 'Project', 'Value': 'SecureMLOps'},
                    {'Key': 'Environment', 'Value': 'Production'}
                ]
            )
            logger.info(f"Created model package group: {group_name}")
            return response['ModelPackageGroupArn']

        except self.sm_client.exceptions.ResourceInUse:
            logger.info(f"Model package group {group_name} already exists")
            response = self.sm_client.describe_model_package_group(
                ModelPackageGroupName=group_name
            )
            return response['ModelPackageGroupArn']

    def register_model(
        self,
        model_package_group_name: str,
        model_data_url: str,
        image_uri: str,
        model_metrics: Dict,
        inference_spec: Dict = None,
        approval_status: str = "PendingManualApproval",
        model_card: Optional[Dict] = None
    ) -> str:
        """
        Register a model version in the model registry

        Args:
            model_package_group_name: Name of the model package group
            model_data_url: S3 URL to model artifacts
            image_uri: Container image URI
            model_metrics: Dictionary of model metrics
            inference_spec: Inference specification
            approval_status: Approval status (PendingManualApproval, Approved, Rejected)
            model_card: Model card information

        Returns:
            Model package ARN
        """
        logger.info(f"Registering model in group: {model_package_group_name}")

        # Ensure model package group exists
        self.create_model_package_group(model_package_group_name)

        # Prepare model metrics
        metrics_source = MetricsSource(
            content_type="application/json",
            s3_uri=self._save_metrics_to_s3(model_metrics)
        )

        model_metrics_obj = ModelMetrics(
            model_statistics=metrics_source
        )

        # Default inference spec if not provided
        if inference_spec is None:
            inference_spec = {
                'InferenceSpecification': {
                    'Containers': [{
                        'Image': image_uri,
                        'ModelDataUrl': model_data_url,
                        'Environment': {
                            'SAGEMAKER_PROGRAM': 'inference.py',
                            'SAGEMAKER_SUBMIT_DIRECTORY': model_data_url
                        }
                    }],
                    'SupportedContentTypes': ['application/json', 'text/plain'],
                    'SupportedResponseMIMETypes': ['application/json'],
                    'SupportedRealtimeInferenceInstanceTypes': [
                        'ml.g4dn.xlarge',
                        'ml.g5.xlarge',
                        'ml.g5.2xlarge'
                    ],
                }
            }

        # Create model package
        try:
            response = self.sm_client.create_model_package(
                ModelPackageGroupName=model_package_group_name,
                ModelPackageDescription=f"Model version created at {datetime.now().isoformat()}",
                ModelApprovalStatus=approval_status,
                **inference_spec,
                ModelMetrics={
                    'ModelQuality': {
                        'Statistics': {
                            'ContentType': 'application/json',
                            'S3Uri': self._save_metrics_to_s3(model_metrics)
                        }
                    }
                },
                Tags=[
                    {'Key': 'Perplexity', 'Value': str(model_metrics.get('perplexity', 'N/A'))},
                    {'Key': 'EvalLoss', 'Value': str(model_metrics.get('eval_loss', 'N/A'))},
                    {'Key': 'CreatedAt', 'Value': datetime.now().isoformat()}
                ]
            )

            model_package_arn = response['ModelPackageArn']
            logger.info(f"✓ Model registered: {model_package_arn}")

            # Create model card if provided
            if model_card:
                self._create_model_card(model_package_arn, model_card)

            return model_package_arn

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    def _save_metrics_to_s3(self, metrics: Dict) -> str:
        """Save metrics to S3 and return URI"""
        import tempfile
        import os

        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(metrics, f, indent=2)
            temp_path = f.name

        try:
            # Upload to S3
            bucket = self.session.default_bucket()
            key = f"model-metrics/{datetime.now().strftime('%Y/%m/%d')}/metrics-{datetime.now().timestamp()}.json"

            s3_client = boto3.client('s3')
            s3_client.upload_file(
                temp_path,
                bucket,
                key,
                ExtraArgs={'ServerSideEncryption': 'aws:kms'}
            )

            s3_uri = f"s3://{bucket}/{key}"
            logger.info(f"Metrics saved to {s3_uri}")
            return s3_uri

        finally:
            os.unlink(temp_path)

    def _create_model_card(self, model_package_arn: str, model_card: Dict):
        """Create a model card for governance and documentation"""
        try:
            # Model cards require additional setup
            # This is a placeholder for model card creation
            logger.info("Model card creation - ensure Model Cards are enabled in your account")

        except Exception as e:
            logger.warning(f"Could not create model card: {e}")

    def list_model_versions(
        self,
        model_package_group_name: str,
        approval_status: Optional[str] = None
    ) -> List[Dict]:
        """List all versions of a model"""
        try:
            kwargs = {
                'ModelPackageGroupName': model_package_group_name,
                'SortBy': 'CreationTime',
                'SortOrder': 'Descending'
            }

            if approval_status:
                kwargs['ModelApprovalStatus'] = approval_status

            response = self.sm_client.list_model_packages(**kwargs)

            versions = []
            for package in response.get('ModelPackageSummaryList', []):
                versions.append({
                    'arn': package['ModelPackageArn'],
                    'version': package.get('ModelPackageVersion'),
                    'status': package['ModelApprovalStatus'],
                    'created': package['CreationTime']
                })

            logger.info(f"Found {len(versions)} model versions")
            return versions

        except Exception as e:
            logger.error(f"Failed to list model versions: {e}")
            return []

    def update_approval_status(
        self,
        model_package_arn: str,
        approval_status: str,
        approval_description: str = ""
    ):
        """
        Update the approval status of a model

        Args:
            model_package_arn: ARN of the model package
            approval_status: New status (Approved, Rejected, PendingManualApproval)
            approval_description: Description of the approval decision
        """
        try:
            self.sm_client.update_model_package(
                ModelPackageArn=model_package_arn,
                ModelApprovalStatus=approval_status,
                ApprovalDescription=approval_description
            )

            logger.info(
                f"✓ Updated model approval status to {approval_status}: "
                f"{model_package_arn}"
            )

        except Exception as e:
            logger.error(f"Failed to update approval status: {e}")
            raise

    def get_latest_approved_model(
        self,
        model_package_group_name: str
    ) -> Optional[str]:
        """Get the latest approved model version"""
        versions = self.list_model_versions(
            model_package_group_name,
            approval_status='Approved'
        )

        if versions:
            return versions[0]['arn']
        else:
            logger.warning("No approved model versions found")
            return None

    def get_model_metrics(self, model_package_arn: str) -> Dict:
        """Retrieve metrics for a specific model version"""
        try:
            response = self.sm_client.describe_model_package(
                ModelPackageName=model_package_arn
            )

            # Extract metrics from tags
            tags_response = self.sm_client.list_tags(
                ResourceArn=model_package_arn
            )

            metrics = {}
            for tag in tags_response.get('Tags', []):
                if tag['Key'] in ['Perplexity', 'EvalLoss']:
                    try:
                        metrics[tag['Key'].lower()] = float(tag['Value'])
                    except ValueError:
                        metrics[tag['Key'].lower()] = tag['Value']

            return metrics

        except Exception as e:
            logger.error(f"Failed to get model metrics: {e}")
            return {}


if __name__ == "__main__":
    # Example usage
    registry = SecureModelRegistry()

    # Create model package group
    group_name = "secure-llm-models"
    registry.create_model_package_group(group_name)

    # List versions
    versions = registry.list_model_versions(group_name)
    for v in versions:
        print(f"Version {v['version']}: {v['status']} (created {v['created']})")
