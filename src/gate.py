import logging
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import sys

from constants import METRICS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def find_evaluation_report(eval_dir: str) -> Optional[Path]:
    """
    Search for an evaluation report file (commonly named evaluation_report.json or metrics.json)
    inside the given eval_dir.
    Returns the Path if found, otherwise None.
    """
    eval_path = Path(eval_dir)
    if not eval_path.exists() or not eval_path.is_dir():
        logger.error(f"Evaluation directory not found: {eval_dir}")
        return None

    # Common filenames
    candidates = ["evaluation_report.json", "metrics.json"]
    for name in candidates:
        candidate_path = eval_path / name
        if candidate_path.exists():
            return candidate_path

    logger.error(f"No evaluation report JSON found in directory: {eval_dir}")
    return None


def gate_cli(eval_dir: str, metric: str, threshold: float):
    """
    Gating function that looks for an evaluation report file, reads it, and compares
    a given metric against a threshold.
    """
    report_path = find_evaluation_report(eval_dir)
    if not report_path:
        logger.error(f"Failed to find evaluation report JSON in {eval_dir}")
        return False

    try:
        report = json.loads(report_path.read_text())
    except Exception as e:
        logger.error(f"Failed to parse evaluation report: {e}")
        return False

    # Extract the metric value (simple version)
    value = report[METRICS].get(metric)

    if value is None:
        logger.error(f"Metric {metric} not found in evaluation report.")
        return False

    logger.info(f"Metric {metric} = {value} (threshold = {threshold})")
    if value < threshold:
        logger.warning(f"Gate failed: metric {metric} below threshold {threshold}.")
        return False

    logger.info(f"Gate passed")
    return True


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description="Gate a model evaluation by comparing a chosen metric to a threshold.",
        prog="gate",
        usage="%(prog)s [options]",
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Directory containing evaluation report JSON files (e.g., evaluation_report.json).",
    )

    parser.add_argument(
        "--metric",
        help="Name of the metric to check in the evaluation report (supports dot-notation for nested metrics).",
        default="accuracy",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        help="Threshold value for the metric; gate fails if the metric is below this value.",
        default=0.65,
    )

    args = parser.parse_args(sys_argv)

    gate_cli(**vars(args))


if __name__ == "__main__":
    cli(sys.argv[1:])
