import logging
import os

os.makedirs("logs", exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers when Streamlit reruns
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        for handler in (logging.StreamHandler(), logging.FileHandler("logs/app.log")):
            handler.setFormatter(fmt)
            logger.addHandler(handler)
    return logger
