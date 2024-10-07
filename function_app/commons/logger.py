"""This module is for adding logging messages."""
import function_app.commons.globals as gl
import logging
import os

logger = logging.getLogger("function_app")
if gl.DEBUG_LOG:
    level = logging.getLevelName("INFO")
else:
    level = logging.getLevelName("WARNING")
logger.setLevel(level)

log_formatter = logging.Formatter("%(asctime)s %(levelname)-10s %(filename)s:%(lineno)d %(funcName)20s() :: %(message)s")
# Configure a console output
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logger.addHandler(console_handler)
logger.propagate = False

