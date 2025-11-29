# logging 
import logging

# create logger class

class Logger(name="", level=logging.INFO):
    ''' 
    A simple logger class to encapsulate logging functionality.
    args: 
        name: Name of the logger
        level: Logging level (default: INFO)
    returns: logger instance
    raises: None
    '''
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self._setup_handler()

    def _setup_handler(self):
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def get_logger(self):
        return self.logger
    


# Usage example:
# logger = Logger(__name__).get_logger()
# logger.info("This is an info message")
