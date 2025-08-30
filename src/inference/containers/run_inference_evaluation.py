#!/usr/bin/env python3
"""
Script to run regression inference on FOMO Task3 preprocessed data and evaluate results using a script folder.

This script:
1. Discovers all subjects and sessions with T1 and T2 modalities
2. Runs the regression script from a specified folder on each subject/session
3. Collects predictions and ground truth labels
4. Computes MAE and correlation metrics
5. Saves detailed results
"""

import os
import sys
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

class ScriptInferenceEvaluator(BaseInferenceEvaluator):
    """Class to handle regression inference and evaluation on FOMO Task3 data using a script folder."""

    def __init__(self,
                 data_root: str,
                 labels_root: str,
                 script_folder: str,
                 output_dir: str,
                 max_workers: int = 4):
        """
        Initialize the evaluator.

        Args:
            data_root: Path to preprocessed data directory
            labels_root: Path to labels directory
            script_folder: Path to folder containing the inference script
            output_dir: Directory to save outputs
            max_workers: Maximum number of parallel workers
        """
        super().__init__(data_root, labels_root, output_dir, max_workers)
        self.script_folder = Path(script_folder)

        # Locate the predict script in the script folder
        self.predict_script = self.script_folder / "predict.py"

        if not self.predict_script.exists():
            raise FileNotFoundError(f"Predict script not found: {self.predict_script}")

    def run_inference(self, subject_id: str, session_id: str) -> Tuple[str, str, float, float, bool]:
        """
        Run inference on a single subject/session using the script.

        Args:
            subject_id: Subject identifier
            session_id: Session identifier

        Returns:
            Tuple of (subject_id, session_id, predicted_age, ground_truth_age, success)
        """
        try:
            # Define paths
            data_session_dir = self.data_root / subject_id / session_id
            t1_path = data_session_dir / "t1.nii.gz"
            t2_path = data_session_dir / "t2.nii.gz"

            # Create temporary output file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
                output_path = tmp_file.name

            try:
                # Construct python command to run the predict.py script from script_folder
                cmd = [
                    sys.executable,  # Use the same Python interpreter
                    str(self.predict_script),
                    "--t1", str(t1_path),
                    "--t2", str(t2_path),
                    "--output", output_path,
                ]

                logger.info(f"Running inference for {subject_id}/{session_id}")
                logger.debug(f"Command: {' '.join(cmd)}")

                # Set up environment for the script
                env = os.environ.copy()
                env['PYTHONPATH'] = str(self.script_folder) + ":" + env.get('PYTHONPATH', '')

                # Run the script
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                    env=env,
                    cwd=str(self.script_folder)
                )

                if result.returncode != 0:
                    logger.error(f"Script failed for {subject_id}/{session_id}:")
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
        description="Run regression inference evaluation on FOMO Task3 data using a script folder"
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
        "--script_folder",
        type=str,
        default="/home/mg873uh/Projects_kb/baseline-codebase/regression_container/app",
        help="Path to folder containing the inference script"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/mg873uh/Projects_kb/baseline-codebase/src/inference/containers/evaluation_results",
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
    evaluator = ScriptInferenceEvaluator(
        data_root=args.data_root,
        labels_root=args.labels_root,
        script_folder=args.script_folder,
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
