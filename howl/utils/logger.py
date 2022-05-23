import logging
import os
import sys
from logging.handlers import RotatingFileHandler

import coloredlogs


def setup_logger(name, level=logging.INFO, use_stdout=True, log_path=None):
    """Setup of logger

    Args:
        name: Logger name
        level: Verbosity level
        use_stdout: Whether to add stdout as a handler
        log_path: Log file path

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
            log_path, mode="a", maxBytes=5 * 1024 * 1024, backupCount=2, encoding=None, delay=False,
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


class Logger:
    """logging.Logger instance cannot be pickled. This class will manage a single logger across the system"""

    LOGGER = None
    NAME = "Howl"

    def __init__(self):
        raise NotImplementedError("All Logger methods are static. Do not instantiate")

    @staticmethod
    def init_logger_if_missing():
        """Create logger if missing"""
        if Logger.LOGGER is None:
            Logger.LOGGER = setup_logger(Logger.NAME)

    @staticmethod
    def info(msg: str):
        """log for INFO level"""
        Logger.init_logger_if_missing()
        Logger.LOGGER.info(msg)

    @staticmethod
    def debug(msg: str):
        """log for DEBUG level"""
        Logger.init_logger_if_missing()
        Logger.LOGGER.debug(msg)

    @staticmethod
    def warning(msg: str):
        """log for WARNING level"""
        Logger.init_logger_if_missing()
        Logger.LOGGER.warning(msg)

    @staticmethod
    def error(msg: str):
        """log for ERROR level"""
        Logger.init_logger_if_missing()
        Logger.LOGGER.error(msg)
