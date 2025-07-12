import logging
import os
from datetime import datetime


def setup_logger(save_name, log_dir="./log"):
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger()

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f"{log_dir}/{save_name}_{current_time}.log")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_logger(name=None):
    return logging.getLogger(name)
