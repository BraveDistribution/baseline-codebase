#!/usr/bin/env python3
"""
Script to run regression inference on FOMO Task3 preprocessed data and evaluate results using a container.

This script:
1. Discovers all subjects and sessions with T1 and T2 modalities
2. Runs the regression container on each subject/session
3. Collects predictions and ground truth labels
4. Computes MAE and correlation metrics
5. Saves detailed results
"""

import os
import subprocess
import numpy as np
from pathlib import Path
import argparse
from typing import Tuple
import logging
import tempfile
from common import BaseInferenceEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RegressionInferenceEvaluator(BaseInferenceEvaluator):
    """Class to handle regression inference and evaluation on FOMO Task3 data using containers."""

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
        super().__init__(data_root, labels_root, output_dir, max_workers)
        self.container_path = Path(container_path)

    def run_inference(self, subject_id: str, session_id: str) -> Tuple[str, str, float, float, bool]:
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


def main():
    parser = argparse.ArgumentParser(
        description="Run regression inference evaluation on FOMO Task3 data using containers"
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
