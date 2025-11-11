"""
Secure SageMaker Training Script for LLM Fine-tuning
Implements security controls and experiment tracking
"""
import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk, load_dataset
import evaluate

# SageMaker imports
import sagemaker
from sagemaker.experiments import Run

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SecureLLMTrainer:
    """
    Secure trainer for LLM fine-tuning with:
    - Security controls
    - Experiment tracking
    - Model evaluation
    - Threshold checking
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None

        # Security: Validate environment
        self._validate_environment()

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def _validate_environment(self):
        """Validate security settings and environment"""
        logger.info("Validating secure environment...")

        # Check for required environment variables
        required_vars = ['SM_MODEL_DIR', 'SM_OUTPUT_DATA_DIR']
        for var in required_vars:
            if var not in os.environ:
                logger.warning(f"{var} not set, using default")

        # Security: Ensure we're running in SageMaker environment
        if os.path.exists('/opt/ml/input/config/'):
            logger.info("✓ Running in SageMaker environment")
        else:
            logger.warning("Not running in SageMaker, some features may not work")

        # Check for GPU
        if torch.cuda.is_available():
            logger.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("No GPU available, training will be slow")

    def load_model_and_tokenizer(self):
        """Load model and tokenizer with security checks"""
        logger.info(f"Loading model: {self.args.model_name}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_name,
                trust_remote_code=False,  # Security: Don't trust remote code
                use_fast=True
            )

            # Add padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with security settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                trust_remote_code=False,  # Security: Don't trust remote code
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                use_cache=False  # Required for gradient checkpointing
            )

            # Enable gradient checkpointing for memory efficiency
            self.model.gradient_checkpointing_enable()

            logger.info(f"✓ Model loaded successfully")
            logger.info(f"Model size: {self.model.num_parameters():,} parameters")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def setup_lora(self):
        """Configure LoRA for efficient fine-tuning"""
        logger.info("Setting up LoRA configuration...")

        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        logger.info(
            f"✓ LoRA configured - Trainable params: {trainable_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

    def load_datasets(self):
        """Load and prepare datasets"""
        logger.info(f"Loading datasets from {self.args.train_data}")

        try:
            # Load from S3 or local path
            if self.args.train_data.startswith('s3://'):
                # Download from S3 to local
                import boto3
                s3 = boto3.client('s3')
                # Implementation would download data here
                pass

            # Load dataset
            if os.path.exists(self.args.train_data):
                train_data = load_from_disk(self.args.train_data)
            else:
                # Load from HuggingFace dataset
                train_data = load_dataset(self.args.train_data, split='train')

            if self.args.eval_data:
                if os.path.exists(self.args.eval_data):
                    eval_data = load_from_disk(self.args.eval_data)
                else:
                    eval_data = load_dataset(self.args.eval_data, split='validation')
            else:
                # Split train data
                split = train_data.train_test_split(test_size=0.1)
                train_data = split['train']
                eval_data = split['test']

            # Tokenize datasets
            def tokenize_function(examples):
                return self.tokenizer(
                    examples['text'],
                    padding='max_length',
                    truncation=True,
                    max_length=self.args.max_seq_length
                )

            self.train_dataset = train_data.map(
                tokenize_function,
                batched=True,
                remove_columns=train_data.column_names
            )

            self.eval_dataset = eval_data.map(
                tokenize_function,
                batched=True,
                remove_columns=eval_data.column_names
            )

            logger.info(f"✓ Datasets loaded - Train: {len(self.train_dataset)}, Eval: {len(self.eval_dataset)}")

        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            raise

    def train(self, experiment_name: Optional[str] = None, run_name: Optional[str] = None):
        """Train the model with experiment tracking"""
        logger.info("Starting training...")

        # Training arguments with security settings
        training_args = TrainingArguments(
            output_dir=os.environ.get('SM_MODEL_DIR', './model_output'),
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            warmup_steps=self.args.warmup_steps,
            logging_dir=os.environ.get('SM_OUTPUT_DATA_DIR', './logs'),
            logging_steps=self.args.logging_steps,
            eval_steps=self.args.eval_steps,
            save_steps=self.args.save_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to=["tensorboard"],
            disable_tqdm=False,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )

        # Train with experiment tracking
        if experiment_name and run_name:
            with Run(
                experiment_name=experiment_name,
                run_name=run_name,
                sagemaker_session=sagemaker.Session()
            ) as run:
                # Log hyperparameters
                run.log_parameters({
                    "model_name": self.args.model_name,
                    "epochs": self.args.epochs,
                    "batch_size": self.args.batch_size,
                    "learning_rate": self.args.learning_rate,
                    "lora_r": self.args.lora_r,
                    "lora_alpha": self.args.lora_alpha,
                })

                # Train
                train_result = trainer.train()

                # Log metrics
                run.log_metric("train_loss", train_result.training_loss)
        else:
            train_result = trainer.train()

        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)

        logger.info("✓ Training completed")
        return trainer, train_result

    def evaluate(self, trainer) -> Dict:
        """Evaluate the model"""
        logger.info("Evaluating model...")

        eval_results = trainer.evaluate()

        # Calculate perplexity
        eval_results['perplexity'] = torch.exp(torch.tensor(eval_results['eval_loss'])).item()

        logger.info("Evaluation Results:")
        for key, value in eval_results.items():
            logger.info(f"  {key}: {value:.4f}")

        return eval_results

    def check_performance_threshold(self, metrics: Dict) -> bool:
        """Check if metrics meet performance threshold"""
        logger.info("Checking performance thresholds...")

        passed = True

        # Check perplexity
        if metrics.get('perplexity', float('inf')) > self.args.max_perplexity:
            logger.error(
                f"✗ Perplexity {metrics['perplexity']:.2f} exceeds "
                f"threshold {self.args.max_perplexity}"
            )
            passed = False
        else:
            logger.info(
                f"✓ Perplexity {metrics['perplexity']:.2f} meets threshold"
            )

        # Check eval loss
        if metrics.get('eval_loss', float('inf')) > self.args.max_eval_loss:
            logger.error(
                f"✗ Eval loss {metrics['eval_loss']:.4f} exceeds "
                f"threshold {self.args.max_eval_loss}"
            )
            passed = False
        else:
            logger.info(
                f"✓ Eval loss {metrics['eval_loss']:.4f} meets threshold"
            )

        return passed


def parse_args():
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--max_seq_length', type=int, default=512)

    # Training parameters
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--save_steps', type=int, default=500)

    # LoRA parameters
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)

    # Data parameters
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--eval_data', type=str, default=None)

    # Evaluation thresholds
    parser.add_argument('--max_perplexity', type=float, default=20.0)
    parser.add_argument('--max_eval_loss', type=float, default=1.5)

    # Experiment tracking
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Initialize trainer
    trainer = SecureLLMTrainer(args)

    # Load model
    trainer.load_model_and_tokenizer()

    # Setup LoRA
    trainer.setup_lora()

    # Load datasets
    trainer.load_datasets()

    # Train
    hf_trainer, train_result = trainer.train(
        experiment_name=args.experiment_name,
        run_name=args.run_name
    )

    # Evaluate
    eval_metrics = trainer.evaluate(hf_trainer)

    # Check thresholds
    if trainer.check_performance_threshold(eval_metrics):
        logger.info("✓ Model meets all performance thresholds")
        exit(0)
    else:
        logger.error("✗ Model does not meet performance thresholds")
        exit(1)
