"""Utility functions for input sanitization and security."""

import re
from typing import Optional


def sanitize_input(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        text: Input text to sanitize
        max_length: Optional maximum length
    
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Remove potentially dangerous characters and SQL keywords
    sanitized = re.sub(r'[<>"\';\\]', '', text)
    # Remove SQL injection patterns
    sql_patterns = [
        r'(?i)\bDROP\s+TABLE\b',
        r'(?i)\bDELETE\s+FROM\b',
        r'(?i)\bINSERT\s+INTO\b',
        r'(?i)\bUPDATE\s+SET\b',
        r'(?i)\bSELECT\s+.*\s+FROM\b',
        r'(?i)\bUNION\s+SELECT\b',
        r'--',  # SQL comments
        r'/\*.*?\*/',  # SQL block comments
    ]
    for pattern in sql_patterns:
        sanitized = re.sub(pattern, '', sanitized)
    sanitized = sanitized.strip()
    
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized


def validate_email(email: str) -> bool:
    """
    Validate email format.
    
    Args:
        email: Email address to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not email:
        return False
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email))


def sanitize_product_name(name: str) -> str:
    """
    Sanitize product name for database queries.
    
    Args:
        name: Product name
    
    Returns:
        Sanitized product name
    """
    return sanitize_input(name, max_length=200)


def sanitize_customer_name(name: str) -> str:
    """
    Sanitize customer name.
    
    Args:
        name: Customer name
    
    Returns:
        Sanitized customer name
    """
    return sanitize_input(name, max_length=100)

