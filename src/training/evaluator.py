"""
Model Evaluation Module with Performance Threshold Checking
"""
import logging
from typing import Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import evaluate
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation with industry best practices
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str = None,
        device: str = "auto"
    ):
        """Initialize evaluator with model and tokenizer"""
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device if device == "auto" else None
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device != "cpu" else "cpu"
        )

        if device != "auto":
            self.model.to(self.device)

        self.model.eval()
        logger.info(f"Model loaded on {self.device}")

    def calculate_perplexity(
        self,
        dataset,
        batch_size: int = 8,
        max_length: int = 512
    ) -> float:
        """
        Calculate perplexity on a dataset
        Industry benchmark: < 20 for good models, < 10 for excellent models
        """
        logger.info("Calculating perplexity...")

        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]

                # Tokenize
                encodings = self.tokenizer(
                    batch['text'] if 'text' in batch else batch,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(self.device)

                # Forward pass
                outputs = self.model(**encodings, labels=encodings['input_ids'])

                # Calculate loss
                total_loss += outputs.loss.item() * encodings['input_ids'].numel()
                total_tokens += encodings['input_ids'].numel()

                if (i // batch_size) % 10 == 0:
                    logger.info(f"Processed {i}/{len(dataset)} samples")

        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        logger.info(f"Perplexity: {perplexity:.2f}")
        return perplexity

    def evaluate_generation_quality(
        self,
        test_prompts: List[str],
        max_new_tokens: int = 100
    ) -> Dict:
        """
        Evaluate generation quality using multiple metrics
        """
        logger.info("Evaluating generation quality...")

        generations = []
        for prompt in test_prompts:
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generations.append(generated_text)

        return {
            'generations': generations,
            'num_samples': len(generations)
        }

    def evaluate_metrics(
        self,
        eval_dataset,
        reference_dataset=None
    ) -> Dict:
        """
        Comprehensive evaluation with multiple metrics
        """
        metrics = {}

        # 1. Perplexity (lower is better)
        perplexity = self.calculate_perplexity(eval_dataset)
        metrics['perplexity'] = perplexity

        # 2. Loss
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for i in range(0, min(len(eval_dataset), 1000), 8):
                batch = eval_dataset[i:i + 8]
                encodings = self.tokenizer(
                    batch['text'] if 'text' in batch else batch,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                outputs = self.model(**encodings, labels=encodings['input_ids'])
                total_loss += outputs.loss.item()
                num_batches += 1

        metrics['eval_loss'] = total_loss / num_batches if num_batches > 0 else float('inf')

        # 3. Token accuracy (if applicable)
        metrics['token_count'] = len(self.tokenizer)

        return metrics

    def check_thresholds(
        self,
        metrics: Dict,
        thresholds: Dict = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if metrics meet performance thresholds
        Industry best practices:
        - Perplexity: < 20 (good), < 10 (excellent)
        - Eval Loss: < 1.5 (good), < 1.0 (excellent)
        """
        if thresholds is None:
            thresholds = {
                'perplexity_max': 20.0,
                'eval_loss_max': 1.5,
            }

        passed = True
        failures = []

        # Check perplexity
        if metrics.get('perplexity', float('inf')) > thresholds['perplexity_max']:
            passed = False
            failures.append(
                f"Perplexity {metrics['perplexity']:.2f} exceeds "
                f"threshold {thresholds['perplexity_max']}"
            )

        # Check eval loss
        if metrics.get('eval_loss', float('inf')) > thresholds['eval_loss_max']:
            passed = False
            failures.append(
                f"Eval loss {metrics['eval_loss']:.4f} exceeds "
                f"threshold {thresholds['eval_loss_max']}"
            )

        if passed:
            logger.info("✓ All performance thresholds met")
        else:
            logger.error("✗ Performance thresholds not met:")
            for failure in failures:
                logger.error(f"  - {failure}")

        return passed, failures

    def generate_evaluation_report(
        self,
        metrics: Dict,
        threshold_check: Tuple[bool, List[str]],
        output_path: str = "evaluation_report.json"
    ):
        """Generate comprehensive evaluation report"""
        import json

        report = {
            'metrics': metrics,
            'thresholds_passed': threshold_check[0],
            'threshold_failures': threshold_check[1],
            'model_path': self.model_path,
            'device': str(self.device)
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Evaluation report saved to {output_path}")
        return report


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--eval_data', type=str, required=True)
    parser.add_argument('--output_report', type=str, default='evaluation_report.json')
    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ModelEvaluator(args.model_path)

    # Load eval dataset
    if args.eval_data.startswith('s3://'):
        # Handle S3 path
        pass
    else:
        eval_dataset = load_from_disk(args.eval_data)

    # Evaluate
    metrics = evaluator.evaluate_metrics(eval_dataset)

    # Check thresholds
    threshold_check = evaluator.check_thresholds(metrics)

    # Generate report
    evaluator.generate_evaluation_report(
        metrics,
        threshold_check,
        args.output_report
    )
