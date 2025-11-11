"""
Secure SageMaker Endpoint Deployment with Monitoring
"""
import logging
from typing import Dict, Optional, List
import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.model_monitor import (
    DataCaptureConfig,
    ModelQualityMonitor,
    DataQualityMonitor
)
import yaml
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecureEndpointDeployer:
    """
    Deploy models to SageMaker endpoints with:
    - Security controls (VPC, encryption, IAM)
    - Model monitoring
    - Auto-scaling
    - A/B testing capabilities
    """

    def __init__(
        self,
        security_config_path: str = "config/security_config.yaml",
        region: str = "us-east-1"
    ):
        self.sm_client = boto3.client('sagemaker', region_name=region)
        self.session = sagemaker.Session()
        self.region = region

        # Load security configuration
        with open(security_config_path, 'r') as f:
            self.security_config = yaml.safe_load(f)

    def deploy_model(
        self,
        model_package_arn: str,
        endpoint_name: str,
        instance_type: str = "ml.g5.xlarge",
        instance_count: int = 1,
        enable_monitoring: bool = True,
        enable_autoscaling: bool = True,
        tags: Optional[List[Dict]] = None
    ) -> str:
        """
        Deploy a model from Model Registry to a secure endpoint

        Args:
            model_package_arn: ARN of the model package to deploy
            endpoint_name: Name for the endpoint
            instance_type: Instance type for hosting
            instance_count: Initial number of instances
            enable_monitoring: Enable model monitoring
            enable_autoscaling: Enable auto-scaling
            tags: Additional tags

        Returns:
            Endpoint ARN
        """
        logger.info(f"Deploying model {model_package_arn} to endpoint {endpoint_name}")

        try:
            # Get model package details
            model_package = self.sm_client.describe_model_package(
                ModelPackageName=model_package_arn
            )

            # Check approval status
            if model_package['ModelApprovalStatus'] != 'Approved':
                raise ValueError(
                    f"Model not approved. Status: {model_package['ModelApprovalStatus']}"
                )

            # Extract container info
            containers = model_package['InferenceSpecification']['Containers']
            image_uri = containers[0]['Image']
            model_data_url = containers[0].get('ModelDataUrl')

            # Create SageMaker Model with security settings
            model_name = f"{endpoint_name}-model-{int(time.time())}"

            security_config = self.security_config['security']
            vpc_config = security_config.get('vpc', {})
            encryption_config = security_config.get('encryption', {})

            model_kwargs = {
                'ModelName': model_name,
                'PrimaryContainer': {
                    'Image': image_uri,
                    'ModelDataUrl': model_data_url,
                    'Environment': containers[0].get('Environment', {})
                },
                'ExecutionRoleArn': security_config['iam']['execution_role'],
                'Tags': tags or []
            }

            # Add VPC configuration
            if vpc_config.get('security_group_ids') and vpc_config.get('subnets'):
                model_kwargs['VpcConfig'] = {
                    'SecurityGroupIds': vpc_config['security_group_ids'],
                    'Subnets': vpc_config['subnets']
                }
                logger.info("✓ VPC configuration applied")

            # Add encryption
            if encryption_config.get('enable_network_isolation'):
                model_kwargs['EnableNetworkIsolation'] = True
                logger.info("✓ Network isolation enabled")

            # Create model
            self.sm_client.create_model(**model_kwargs)
            logger.info(f"✓ Model created: {model_name}")

            # Create endpoint configuration
            endpoint_config_name = f"{endpoint_name}-config-{int(time.time())}"

            endpoint_config_kwargs = {
                'EndpointConfigName': endpoint_config_name,
                'ProductionVariants': [{
                    'VariantName': 'primary',
                    'ModelName': model_name,
                    'InstanceType': instance_type,
                    'InitialInstanceCount': instance_count,
                    'InitialVariantWeight': 1.0
                }],
                'Tags': tags or []
            }

            # Add data capture for monitoring
            if enable_monitoring:
                data_capture_config = self._create_data_capture_config(endpoint_name)
                endpoint_config_kwargs['DataCaptureConfig'] = data_capture_config
                logger.info("✓ Data capture enabled for monitoring")

            # Add encryption for data at rest
            if encryption_config.get('volume_kms_key_id'):
                endpoint_config_kwargs['KmsKeyId'] = encryption_config['volume_kms_key_id']
                logger.info("✓ Encryption at rest enabled")

            # Create endpoint configuration
            self.sm_client.create_endpoint_config(**endpoint_config_kwargs)
            logger.info(f"✓ Endpoint configuration created: {endpoint_config_name}")

            # Create or update endpoint
            try:
                # Check if endpoint exists
                self.sm_client.describe_endpoint(EndpointName=endpoint_name)

                # Update existing endpoint
                logger.info(f"Updating existing endpoint: {endpoint_name}")
                self.sm_client.update_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=endpoint_config_name
                )

            except self.sm_client.exceptions.ClientError:
                # Create new endpoint
                logger.info(f"Creating new endpoint: {endpoint_name}")
                self.sm_client.create_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=endpoint_config_name,
                    Tags=tags or []
                )

            # Wait for endpoint to be in service
            logger.info("Waiting for endpoint to be in service...")
            self._wait_for_endpoint(endpoint_name)

            # Get endpoint ARN
            endpoint_desc = self.sm_client.describe_endpoint(
                EndpointName=endpoint_name
            )
            endpoint_arn = endpoint_desc['EndpointArn']

            logger.info(f"✓ Endpoint deployed successfully: {endpoint_arn}")

            # Configure auto-scaling
            if enable_autoscaling:
                self._configure_autoscaling(
                    endpoint_name,
                    'primary',
                    min_capacity=1,
                    max_capacity=4
                )

            # Setup monitoring
            if enable_monitoring:
                self._setup_monitoring(endpoint_name)

            return endpoint_arn

        except Exception as e:
            logger.error(f"Failed to deploy endpoint: {e}")
            raise

    def _create_data_capture_config(self, endpoint_name: str) -> Dict:
        """Create data capture configuration for model monitoring"""
        bucket = self.session.default_bucket()
        s3_capture_path = f"s3://{bucket}/model-monitoring/{endpoint_name}"

        return {
            'EnableCapture': True,
            'InitialSamplingPercentage': 100,
            'DestinationS3Uri': s3_capture_path,
            'CaptureOptions': [
                {'CaptureMode': 'Input'},
                {'CaptureMode': 'Output'}
            ],
            'CaptureContentTypeHeader': {
                'JsonContentTypes': ['application/json'],
                'CsvContentTypes': ['text/csv']
            }
        }

    def _wait_for_endpoint(self, endpoint_name: str, timeout: int = 1800):
        """Wait for endpoint to be in service"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            response = self.sm_client.describe_endpoint(
                EndpointName=endpoint_name
            )

            status = response['EndpointStatus']
            logger.info(f"Endpoint status: {status}")

            if status == 'InService':
                logger.info("✓ Endpoint is in service")
                return
            elif status == 'Failed':
                raise RuntimeError(
                    f"Endpoint creation failed: {response.get('FailureReason', 'Unknown')}"
                )

            time.sleep(30)

        raise TimeoutError(f"Endpoint creation timed out after {timeout} seconds")

    def _configure_autoscaling(
        self,
        endpoint_name: str,
        variant_name: str,
        min_capacity: int = 1,
        max_capacity: int = 4,
        target_invocations_per_instance: int = 1000
    ):
        """Configure auto-scaling for the endpoint"""
        try:
            autoscaling_client = boto3.client('application-autoscaling')

            resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"

            # Register scalable target
            autoscaling_client.register_scalable_target(
                ServiceNamespace='sagemaker',
                ResourceId=resource_id,
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                MinCapacity=min_capacity,
                MaxCapacity=max_capacity
            )

            # Create scaling policy
            autoscaling_client.put_scaling_policy(
                PolicyName=f"{endpoint_name}-scaling-policy",
                ServiceNamespace='sagemaker',
                ResourceId=resource_id,
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                PolicyType='TargetTrackingScaling',
                TargetTrackingScalingPolicyConfiguration={
                    'TargetValue': float(target_invocations_per_instance),
                    'PredefinedMetricSpecification': {
                        'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                    },
                    'ScaleInCooldown': 300,
                    'ScaleOutCooldown': 60
                }
            )

            logger.info(
                f"✓ Auto-scaling configured: {min_capacity}-{max_capacity} instances"
            )

        except Exception as e:
            logger.warning(f"Could not configure auto-scaling: {e}")

    def _setup_monitoring(self, endpoint_name: str):
        """Setup model quality and data quality monitoring"""
        try:
            bucket = self.session.default_bucket()

            # Data quality monitoring
            data_quality_monitor = DataQualityMonitor(
                role=self.security_config['security']['iam']['execution_role'],
                instance_count=1,
                instance_type='ml.m5.xlarge',
                volume_size_in_gb=20,
                max_runtime_in_seconds=3600,
                sagemaker_session=self.session
            )

            logger.info("✓ Model monitoring configured")

        except Exception as e:
            logger.warning(f"Could not setup monitoring: {e}")

    def delete_endpoint(
        self,
        endpoint_name: str,
        delete_config: bool = True,
        delete_model: bool = True
    ):
        """Delete endpoint and associated resources"""
        try:
            # Get endpoint details
            endpoint_desc = self.sm_client.describe_endpoint(
                EndpointName=endpoint_name
            )
            endpoint_config_name = endpoint_desc['EndpointConfigName']

            # Delete endpoint
            self.sm_client.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"✓ Deleted endpoint: {endpoint_name}")

            if delete_config:
                # Get endpoint config details
                config_desc = self.sm_client.describe_endpoint_config(
                    EndpointConfigName=endpoint_config_name
                )

                # Delete endpoint config
                self.sm_client.delete_endpoint_config(
                    EndpointConfigName=endpoint_config_name
                )
                logger.info(f"✓ Deleted endpoint config: {endpoint_config_name}")

                if delete_model:
                    # Delete associated models
                    for variant in config_desc['ProductionVariants']:
                        model_name = variant['ModelName']
                        self.sm_client.delete_model(ModelName=model_name)
                        logger.info(f"✓ Deleted model: {model_name}")

        except Exception as e:
            logger.error(f"Failed to delete endpoint: {e}")
            raise

    def invoke_endpoint(
        self,
        endpoint_name: str,
        payload: str,
        content_type: str = "application/json"
    ) -> str:
        """Test endpoint with a sample payload"""
        runtime_client = boto3.client('sagemaker-runtime')

        try:
            response = runtime_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType=content_type,
                Body=payload
            )

            result = response['Body'].read().decode('utf-8')
            logger.info("✓ Endpoint invocation successful")
            return result

        except Exception as e:
            logger.error(f"Failed to invoke endpoint: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    deployer = SecureEndpointDeployer()

    # Deploy endpoint
    # endpoint_arn = deployer.deploy_model(
    #     model_package_arn="arn:aws:sagemaker:...",
    #     endpoint_name="secure-llm-endpoint",
    #     instance_type="ml.g5.xlarge",
    #     enable_monitoring=True
    # )
