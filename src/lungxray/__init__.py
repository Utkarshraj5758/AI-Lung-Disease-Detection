"""Pneumonia vs normal chest X-ray classification with Grad-CAM. Research use only."""

import logging

__version__ = "1.0.0"


def configure_logging(level: int = logging.INFO) -> None:
    """Send ``lungxray.*`` log records to stderr once per process (app and API servers)."""
    package_log = logging.getLogger("lungxray")
    if not package_log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        package_log.addHandler(handler)
        package_log.setLevel(level)
