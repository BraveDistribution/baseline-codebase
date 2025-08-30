#!/usr/bin/env python3
"""
Script to run regression inference on FOMO Task3 preprocessed data and evaluate results.

This script:
1. Discovers all subjects and sessions with T1 and T2 modalities
2. Runs the regression container on each subject/session
3. Collects predictions and ground truth labels
4. Computes MAE and correlation metrics
5. Saves detailed results
"""

import os
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from scipy.stats import pearsonr
from typing import List, Tuple, Dict
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RegressionInferenceEvaluator:
    """Class to handle regression inference and evaluation on FOMO Task3 data."""

    def __init__(self,
                 data_root: str,
                 labels_root: str,
                 container_path: str,
                 output_dir: str,
                 max_workers: int = 4):
        """
        Initialize the evaluator.

        Args:
            data_root: Path to preprocessed data directory
            labels_root: Path to labels directory
            container_path: Path to regression.sif container
            output_dir: Directory to save outputs
            max_workers: Maximum number of parallel workers
        """
        self.data_root = Path(data_root)
        self.labels_root = Path(labels_root)
        self.container_path = Path(container_path)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers

        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Setup results storage
        self.results = []

    def discover_subjects_sessions(self) -> List[Tuple[str, str]]:
        """
        Discover all valid subject/session pairs that have both T1 and T2 modalities.

        Returns:
            List of (subject_id, session_id) tuples
        """
        valid_pairs = []

        # Get all subject directories
        for subject_dir in sorted(self.data_root.glob("sub_*")):
            if not subject_dir.is_dir():
                continue

            subject_id = subject_dir.name

            # Get all session directories for this subject
            for session_dir in sorted(subject_dir.glob("ses_*")):
                if not session_dir.is_dir():
                    continue

                session_id = session_dir.name

                # Check if both T1 and T2 exist
                t1_path = session_dir / "t1.nii.gz"
                t2_path = session_dir / "t2.nii.gz"

                if t1_path.exists() and t2_path.exists():
                    # Also check if label exists
                    label_path = self.labels_root / subject_id / session_id / "label.txt"
                    if label_path.exists():
                        valid_pairs.append((subject_id, session_id))
                    else:
                        logger.warning(f"Label not found for {subject_id}/{session_id}")
                else:
                    logger.warning(f"Missing modalities for {subject_id}/{session_id}")

        logger.info(f"Found {len(valid_pairs)} valid subject/session pairs")
        return valid_pairs

    def load_ground_truth_label(self, subject_id: str, session_id: str) -> float:
        """
        Load ground truth label for a subject/session.

        Args:
            subject_id: Subject identifier
            session_id: Session identifier

        Returns:
            Ground truth age as float
        """
        label_path = self.labels_root / subject_id / session_id / "label.txt"
        try:
            with open(label_path, 'r') as f:
                age = float(f.read().strip())
            return age
        except Exception as e:
            logger.error(f"Error loading label for {subject_id}/{session_id}: {e}")
            raise

    def run_container_inference(self, subject_id: str, session_id: str) -> Tuple[str, str, float, float, bool]:
        """
        Run inference on a single subject/session using the container.

        Args:
            subject_id: Subject identifier
            session_id: Session identifier

        Returns:
            Tuple of (subject_id, session_id, predicted_age, ground_truth_age, success)
        """
        try:
            # Define paths
            data_session_dir = self.data_root / subject_id / session_id
            t1_nii = "t1.nii.gz"
            t2_nii = "t2.nii.gz"

            # Create temporary output file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
                output_path = tmp_file.name

            try:
                # Construct apptainer command
                cmd = [
                    "apptainer", "run",
                    "--bind", f"{data_session_dir}:/input:ro",
                    "--bind", f"{Path(output_path).parent}:/output",
                    "--nv",  # Enable NVIDIA GPU support
                    str(self.container_path),
                    "--t1", f"/input/{t1_nii}",
                    "--t2", f"/input/{t2_nii}",
                    "--output", f"/output/{Path(output_path).name}"
                ]

                logger.info(f"Running inference for {subject_id}/{session_id}")
                logger.debug(f"Command: {' '.join(cmd)}")

                # Run the container
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )

                if result.returncode != 0:
                    logger.error(f"Container failed for {subject_id}/{session_id}:")
                    logger.error(f"STDOUT: {result.stdout}")
                    logger.error(f"STDERR: {result.stderr}")
                    return subject_id, session_id, np.nan, np.nan, False

                # Read prediction from output file
                if os.path.exists(output_path):
                    with open(output_path, 'r') as f:
                        predicted_age = float(f.read().strip())
                else:
                    logger.error(f"Output file not created for {subject_id}/{session_id}")
                    return subject_id, session_id, np.nan, np.nan, False

                # Load ground truth
                ground_truth_age = self.load_ground_truth_label(subject_id, session_id)

                logger.info(f"{subject_id}/{session_id}: Predicted={predicted_age:.1f}, GT={ground_truth_age:.1f}")

                return subject_id, session_id, predicted_age, ground_truth_age, True

            finally:
                # Clean up temporary file
                if os.path.exists(output_path):
                    os.unlink(output_path)

        except Exception as e:
            logger.error(f"Error processing {subject_id}/{session_id}: {e}")
            return subject_id, session_id, np.nan, np.nan, False

    def run_parallel_inference(self, subject_session_pairs: List[Tuple[str, str]]) -> None:
        """
        Run inference on all subject/session pairs in parallel.

        Args:
            subject_session_pairs: List of (subject_id, session_id) tuples
        """
        logger.info(f"Starting parallel inference with {self.max_workers} workers")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_pair = {
                executor.submit(self.run_container_inference, subject_id, session_id): (subject_id, session_id)
                for subject_id, session_id in subject_session_pairs
            }

            # Process completed tasks
            for future in as_completed(future_to_pair):
                subject_id, session_id = future_to_pair[future]
                try:
                    result = future.result()
                    self.results.append({
                        'subject_id': result[0],
                        'session_id': result[1],
                        'predicted_age': result[2],
                        'ground_truth_age': result[3],
                        'success': result[4]
                    })
                except Exception as e:
                    logger.error(f"Error getting result for {subject_id}/{session_id}: {e}")
                    self.results.append({
                        'subject_id': subject_id,
                        'session_id': session_id,
                        'predicted_age': np.nan,
                        'ground_truth_age': np.nan,
                        'success': False
                    })

    def run_sequential_inference(self, subject_session_pairs: List[Tuple[str, str]]) -> None:
        """
        Run inference on all subject/session pairs sequentially.

        Args:
            subject_session_pairs: List of (subject_id, session_id) tuples
        """
        logger.info("Starting sequential inference")

        for i, (subject_id, session_id) in enumerate(subject_session_pairs, 1):
            logger.info(f"Processing {i}/{len(subject_session_pairs)}: {subject_id}/{session_id}")

            result = self.run_container_inference(subject_id, session_id)
            self.results.append({
                'subject_id': result[0],
                'session_id': result[1],
                'predicted_age': result[2],
                'ground_truth_age': result[3],
                'success': result[4]
            })

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute evaluation metrics from collected results.

        Returns:
            Dictionary with computed metrics
        """
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)

        # Filter successful predictions
        successful_df = df[df['success'] == True].copy()

        if len(successful_df) == 0:
            logger.error("No successful predictions found!")
            return {}

        # Remove any NaN values
        successful_df = successful_df.dropna(subset=['predicted_age', 'ground_truth_age'])

        if len(successful_df) == 0:
            logger.error("No valid predictions after removing NaN values!")
            return {}

        predicted = successful_df['predicted_age'].values
        ground_truth = successful_df['ground_truth_age'].values

        # Compute metrics
        mae = np.mean(np.abs(predicted - ground_truth))
        mse = np.mean((predicted - ground_truth) ** 2)
        rmse = np.sqrt(mse)

        # Compute correlation
        correlation, p_value = pearsonr(predicted, ground_truth)

        # Compute additional metrics
        bias = np.mean(predicted - ground_truth)
        std_error = np.std(predicted - ground_truth)

        metrics = {
            'n_total': len(df),
            'n_successful': len(successful_df),
            'success_rate': len(successful_df) / len(df),
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'correlation': correlation,
            'correlation_p_value': p_value,
            'bias': bias,
            'std_error': std_error,
            'min_gt_age': ground_truth.min(),
            'max_gt_age': ground_truth.max(),
            'mean_gt_age': ground_truth.mean(),
            'std_gt_age': ground_truth.std(),
            'min_pred_age': predicted.min(),
            'max_pred_age': predicted.max(),
            'mean_pred_age': predicted.mean(),
            'std_pred_age': predicted.std(),
        }

        return metrics

    def save_results(self) -> None:
        """Save detailed results and metrics to files."""
        # Save detailed results
        results_df = pd.DataFrame(self.results)
        results_path = self.output_dir / "detailed_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Detailed results saved to: {results_path}")

        # Compute and save metrics
        metrics = self.compute_metrics()
        if metrics:
            metrics_path = self.output_dir / "evaluation_metrics.txt"
            with open(metrics_path, 'w') as f:
                f.write("FOMO Task3 Regression Evaluation Results\n")
                f.write("=" * 50 + "\n\n")

                f.write(f"Dataset Statistics:\n")
                f.write(f"  Total subjects/sessions: {metrics['n_total']}\n")
                f.write(f"  Successful predictions: {metrics['n_successful']}\n")
                f.write(f"  Success rate: {metrics['success_rate']:.3f}\n\n")

                f.write(f"Ground Truth Age Statistics:\n")
                f.write(f"  Mean ± Std: {metrics['mean_gt_age']:.1f} ± {metrics['std_gt_age']:.1f}\n")
                f.write(f"  Range: [{metrics['min_gt_age']:.1f}, {metrics['max_gt_age']:.1f}]\n\n")

                f.write(f"Predicted Age Statistics:\n")
                f.write(f"  Mean ± Std: {metrics['mean_pred_age']:.1f} ± {metrics['std_pred_age']:.1f}\n")
                f.write(f"  Range: [{metrics['min_pred_age']:.1f}, {metrics['max_pred_age']:.1f}]\n\n")

                f.write(f"Evaluation Metrics:\n")
                f.write(f"  Mean Absolute Error (MAE): {metrics['mae']:.3f} years\n")
                f.write(f"  Root Mean Square Error (RMSE): {metrics['rmse']:.3f} years\n")
                f.write(f"  Pearson Correlation: {metrics['correlation']:.3f} (p={metrics['correlation_p_value']:.2e})\n")
                f.write(f"  Bias: {metrics['bias']:.3f} years\n")
                f.write(f"  Standard Error: {metrics['std_error']:.3f} years\n")

            logger.info(f"Evaluation metrics saved to: {metrics_path}")

            # Print key metrics to console
            print("\n" + "=" * 60)
            print("REGRESSION EVALUATION RESULTS")
            print("=" * 60)
            print(f"Success Rate: {metrics['success_rate']:.1%} ({metrics['n_successful']}/{metrics['n_total']})")
            print(f"Mean Absolute Error (MAE): {metrics['mae']:.3f} years")
            print(f"Pearson Correlation: {metrics['correlation']:.3f}")
            print(f"Root Mean Square Error (RMSE): {metrics['rmse']:.3f} years")
            print(f"Bias: {metrics['bias']:.3f} years")
            print("=" * 60)

    def run_evaluation(self, parallel: bool = True, max_subjects: int = None) -> None:
        """
        Run the complete evaluation pipeline.

        Args:
            parallel: Whether to run inference in parallel
            max_subjects: Maximum number of subjects to process (for testing)
        """
        logger.info("Starting FOMO Task3 regression evaluation")

        # Discover all valid subject/session pairs
        subject_session_pairs = self.discover_subjects_sessions()

        if max_subjects is not None:
            subject_session_pairs = subject_session_pairs[:max_subjects]
            logger.info(f"Limited to first {max_subjects} subjects for testing")

        if not subject_session_pairs:
            logger.error("No valid subject/session pairs found!")
            return

        # Run inference
        start_time = time.time()
        if parallel:
            self.run_parallel_inference(subject_session_pairs)
        else:
            self.run_sequential_inference(subject_session_pairs)

        end_time = time.time()
        logger.info(f"Inference completed in {end_time - start_time:.1f} seconds")

        # Save results and compute metrics
        self.save_results()


def main():
    parser = argparse.ArgumentParser(
        description="Run regression inference evaluation on FOMO Task3 data"
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/mg873uh/Projects_kb/data/finetuning/fomo-task3/preprocessed",
        help="Path to preprocessed data directory"
    )

    parser.add_argument(
        "--labels_root",
        type=str,
        default="/home/mg873uh/Projects_kb/data/finetuning/fomo-task3/labels",
        help="Path to labels directory"
    )

    parser.add_argument(
        "--container_path",
        type=str,
        default="/home/mg873uh/Projects_kb/baseline-codebase/regression_container/test/regression.sif",
        help="Path to regression.sif container"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/mg873uh/Projects_kb/baseline-codebase/regression_container/evaluation_results",
        help="Directory to save evaluation results"
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,  # Conservative default due to GPU usage
        help="Maximum number of parallel workers"
    )

    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run inference sequentially instead of in parallel"
    )

    parser.add_argument(
        "--max_subjects",
        type=int,
        default=None,
        help="Maximum number of subjects to process (for testing)"
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = RegressionInferenceEvaluator(
        data_root=args.data_root,
        labels_root=args.labels_root,
        container_path=args.container_path,
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )

    # Run evaluation
    evaluator.run_evaluation(
        parallel=not args.sequential,
        max_subjects=args.max_subjects
    )


if __name__ == "__main__":
    main()
