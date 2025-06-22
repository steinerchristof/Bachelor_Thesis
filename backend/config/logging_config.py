"""
Simple logging configuration for the backend
"""
import logging
import sys
from pathlib import Path

def setup_logging(
    log_level="INFO",
    log_file=None,
    use_colors=False,
    include_icons=False,
    enable_performance_filter=True
):
    """Setup basic logging configuration"""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger

# Create component-specific loggers
API_LOGGER = logging.getLogger("api")
SEARCH_LOGGER = logging.getLogger("search")
QA_LOGGER = logging.getLogger("qa")