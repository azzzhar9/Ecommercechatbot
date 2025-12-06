"""Enhanced logging configuration with timestamps and session IDs."""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    session_id: Optional[str] = None
) -> logging.Logger:
    """
    Set up enhanced logger with file and console handlers.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (default: logs/chatbot.log)
        session_id: Optional session ID for tracking
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = str(log_dir / "chatbot.log")
    else:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("ecommerce_chatbot")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter with timestamp and session ID
    class SessionFormatter(logging.Formatter):
        """Custom formatter that includes session ID."""
        
        def format(self, record: logging.LogRecord) -> str:
            """Format log record with session ID if available."""
            if not hasattr(record, 'session_id') and session_id:
                record.session_id = session_id
            elif not hasattr(record, 'session_id'):
                record.session_id = "N/A"
            
            # Add ISO timestamp
            record.iso_timestamp = datetime.now().isoformat()
            
            # Format: [timestamp] [session_id] [level] [message]
            return super().format(record)
    
    formatter = SessionFormatter(
        fmt='[%(iso_timestamp)s] [%(session_id)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(session_id: Optional[str] = None) -> logging.Logger:
    """
    Get or create logger instance.
    
    Args:
        session_id: Optional session ID
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger("ecommerce_chatbot")
    if not logger.handlers:
        log_level = os.getenv("LOG_LEVEL", "INFO")
        setup_logger(log_level=log_level, session_id=session_id)
    return logger

