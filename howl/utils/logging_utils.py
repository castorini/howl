import logging
import os
import sys
from logging.handlers import RotatingFileHandler

import coloredlogs


def setup_logger(name, log_path=None, level=logging.DEBUG, use_stdout=True):
    """Setup of logging to file

    Args:
        name: Logger name
        log_path: Log file path
        level: Verbosity level
        use_stdout: Whether to add stdout as a handler

    Returns:
        A logger
    """

    log_format = "%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s"
    formatter = logging.Formatter(log_format)

    coloredlogs.install(level=level, fmt=log_format)
    logger = logging.getLogger(name)

    # Remove existing handlers if there are any
    if logger.hasHandlers():
        logging.warning(f"Removing existing handlers from {name} logger")
        for handler in logger.handlers:
            logger.removeHandler(handler)

    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # Limit log to 5MB
        handler = RotatingFileHandler(
            log_path, mode="a", maxBytes=5 * 1024 * 1024, backupCount=2, encoding=None, delay=False
        )
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logger.addHandler(handler)

    if use_stdout:
        # Don't propagate messages to root (stderr) logger
        logger.propagate = False

        # Show output in stdout
        std_out_handler = logging.StreamHandler(sys.stdout)
        std_out_handler.setFormatter(formatter)
        std_out_handler.setLevel(level)
        logger.addHandler(std_out_handler)

    logger.info(f"Set up logger ({name}), output path: {log_path}")
    return logger
