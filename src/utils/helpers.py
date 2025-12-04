"""Helper utilities"""

import logging
import sys
from pathlib import Path
from typing import List
from datetime import datetime


def setup_logger(
    name: str = 'ml_model_snow',
    log_file: str = None,
    level: str = 'INFO',
    format_string: str = None
) -> logging.Logger:
    """
    Setup logger with file and console handlers

    Args:
        name: Logger name
        log_file: Log file path
        level: Logging level
        format_string: Log format string

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def create_directories(directories: List[str]):
    """
    Create directories if they don't exist

    Args:
        directories: List of directory paths
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Directory created/verified: {directory}")


def get_timestamp(format: str = '%Y%m%d_%H%M%S') -> str:
    """
    Get current timestamp string

    Args:
        format: Datetime format string

    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime(format)


def print_section(title: str, width: int = 60, char: str = '='):
    """
    Print formatted section header

    Args:
        title: Section title
        width: Total width
        char: Character to use for border
    """
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}\n")


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.2f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {secs:.2f}s"


def save_model_metadata(
    model_name: str,
    metrics: dict,
    hyperparameters: dict,
    file_path: str
):
    """
    Save model metadata to JSON

    Args:
        model_name: Name of the model
        metrics: Performance metrics
        hyperparameters: Model hyperparameters
        file_path: Output file path
    """
    import json

    metadata = {
        'model_name': model_name,
        'timestamp': get_timestamp(),
        'metrics': metrics,
        'hyperparameters': hyperparameters
    }

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Model metadata saved to: {file_path}")


def load_model_metadata(file_path: str) -> dict:
    """
    Load model metadata from JSON

    Args:
        file_path: Input file path

    Returns:
        Metadata dictionary
    """
    import json

    with open(file_path, 'r') as f:
        metadata = json.load(f)

    return metadata


if __name__ == "__main__":
    # Example usage
    logger = setup_logger('test_logger', 'test.log', 'INFO')
    logger.info("This is a test log message")

    create_directories(['test_dir1', 'test_dir2/subdir'])

    print_section("Test Section")

    time_str = format_time(3665)
    print(f"Formatted time: {time_str}")

    timestamp = get_timestamp()
    print(f"Timestamp: {timestamp}")
