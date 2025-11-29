import logging
from typing import Optional

class Logger:
    """
    Simple logger wrapper.

    Args:
        name: logger name (default: module name if empty)
        level: logging level (default: logging.INFO)
        stream: whether to add a StreamHandler (default True)
        file: optional path to a file to log to
        force: if True, remove existing handlers from the logger (useful in tests / REPL)

    Usage:
        log = Logger("myapp").get()
        log.info("hello")
    """
    def __init__(
        self,
        name: str = "",
        level: int = logging.INFO,
        stream: bool = True,
        file: Optional[str] = None,
        force: bool = False,
    ):
        self.logger = logging.getLogger(name or __name__)
        if force:
            self.logger.handlers = []

        self.logger.setLevel(level)

        # Only add handlers if none exist to avoid duplicate logs
        if not self.logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            if stream:
                sh = logging.StreamHandler()
                sh.setFormatter(formatter)
                self.logger.addHandler(sh)

            if file:
                fh = logging.FileHandler(file)
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)

    def get(self) -> logging.Logger:
        """Return the underlying logger"""
        return self.logger
