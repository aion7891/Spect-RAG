"""Logging. SpectraFuse is silent by default (like any library); switch messages on with:

    import spectrafuse
    spectrafuse.enable_logging()          # INFO: one line per pipeline step, with its time
    spectrafuse.enable_logging("DEBUG")   # DEBUG: also one line per query
"""

import logging
import time

logger = logging.getLogger("spectrafuse")
logger.addHandler(logging.NullHandler())


def enable_logging(level="INFO"):
    """Print SpectraFuse's log messages to the terminal (stderr)."""
    if not any(getattr(h, "spectrafuse", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
        handler.spectrafuse = True
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def log_step(name, start, **details):
    """Log one finished step: 'chunking: 1.23 s (chunks=8)'. Returns the time now."""
    extra = ", ".join("%s=%s" % item for item in details.items())
    logger.info("%s: %.2f s%s", name, time.time() - start, " (%s)" % extra if extra else "")
    return time.time()
