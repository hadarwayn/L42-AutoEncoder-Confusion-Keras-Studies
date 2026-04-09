"""Path utilities — all paths relative, no hardcoded absolutes."""
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Return the project root directory (two levels up from this file)."""
    return Path(__file__).resolve().parent.parent.parent


def get_results_dir(experiment: str) -> Path:
    """Return and create the results directory for an experiment.

    Args:
        experiment: 'experiment_a' or 'experiment_b'

    Returns:
        Path to results/experiment_X/ (created if missing)
    """
    results = get_project_root() / "results" / experiment
    results.mkdir(parents=True, exist_ok=True)
    return results


def get_weights_dir(experiment: str) -> Path:
    """Return and create the model weights directory for an experiment."""
    weights = get_results_dir(experiment) / "model_weights"
    weights.mkdir(parents=True, exist_ok=True)
    return weights


def get_logs_dir() -> Path:
    """Return and create the logs directory."""
    logs = get_project_root() / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return logs


def get_data_dir() -> Path:
    """Return and create the data directory for downloaded datasets."""
    data = get_project_root() / "data"
    data.mkdir(parents=True, exist_ok=True)
    return data


def get_log_config_path() -> Optional[Path]:
    """Return the log config JSON path if it exists."""
    config = get_project_root() / "logs" / "config" / "log_config.json"
    return config if config.exists() else None
