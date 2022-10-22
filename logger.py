"""Logger configuration"""
import sys
import logging
from logging.handlers import RotatingFileHandler


logger = logging.getLogger("Despeckling SAR images")
logger.setLevel(logging.INFO)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=(
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            "./despeckling_images.log",
            mode="a",
            maxBytes=100 * 1024 * 1024,
            backupCount=2,
            encoding="UTF-8",
        ),
    ),
)
