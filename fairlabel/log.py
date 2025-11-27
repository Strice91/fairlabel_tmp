import logging
from logging import StreamHandler
from logging.handlers import RotatingFileHandler

from fairlabel.config import settings

formatter = logging.Formatter("%(asctime)s : %(levelname)-8s [%(filename)-13s:%(lineno)-3d] %(message)s")

logger = logging.getLogger(__name__)
if settings.logging.stream:
    stream_handler = StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

if settings.logging.file:
    log_size_byte = 1024 * int(settings.logging.size_kb)
    file_handler = RotatingFileHandler(
        settings.logging.path,
        maxBytes=log_size_byte,
        backupCount=5,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

logger.setLevel(getattr(logging, settings.logging.level))
