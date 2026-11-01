#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/logger.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    This module provides a utility function to create and configure a logger.
    The logger can output to both the console and a file, with customizable
    logging levels and formats.

License:
    Proprietary. All rights reserved.
"""

import logging
import os
import sys
from datetime import datetime


def get_logger(
    name: str = "diffusion",
    log_dir: str = "./logs",
    level: str = "INFO",
    to_file: bool = True
) -> logging.Logger:
    """Creates and configures a logger instance.

    The logger outputs to both the console and, optionally, a file.
    Log files are timestamped for uniqueness.

    Args:
        name: Name of the logger.
        log_dir: Directory where log files are stored.
        level: Logging level as a string (e.g., "INFO", "DEBUG").
        to_file: If True, logs are also written to a file.

    Returns:
        Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # If handlers are already attached, return the logger as is
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"
    )

    # Console handler for output to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler for output to a log file (if enabled)
    if to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logfile = os.path.join(log_dir, f"{name}_{timestamp}.log")
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
