import sys
import logging


logger = logging.getLogger("Despeckling SAR images")
logger.setLevel(logging.INFO)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=(logging.StreamHandler(sys.stdout),),
)
