"""
Centralized datetime utilities to ensure consistent handling across the application.
"""
from datetime import datetime
from typing import Union
import logging

logger = logging.getLogger(__name__)


def parse_timestamp(timestamp: Union[str, datetime, int, float]) -> datetime:
    """
    Safely parse various timestamp formats into a naive datetime object.
    
    This function handles multiple timestamp formats that may be returned by
    external APIs (like Ring) or different Python libraries:
    - datetime objects (with or without timezone)
    - ISO 8601 format strings (e.g., "2025-10-25T18:23:19.683Z")
    - Unix timestamps (seconds since epoch, int or float)
    
    Args:
        timestamp: Can be:
            - datetime object (returned as-is, timezone stripped)
            - ISO format string (e.g., "2025-10-25T18:23:19.683Z")
            - Unix timestamp (int or float)
    
    Returns:
        Naive datetime object (no timezone info)
    
    Raises:
        ValueError: If timestamp format is unrecognized or invalid
    
    Examples:
        >>> parse_timestamp("2025-10-25T18:23:19.683Z")
        datetime.datetime(2025, 10, 25, 18, 23, 19, 683000)
        
        >>> parse_timestamp(1761512638.095)
        datetime.datetime(2025, 10, 26, 12, 23, 58, 95000)
        
        >>> dt = datetime.now()
        >>> parse_timestamp(dt) == dt
        True
    """
    if isinstance(timestamp, datetime):
        # Already a datetime - strip timezone if present
        logger.debug(f"Timestamp is datetime object: {timestamp}")
        return timestamp.replace(tzinfo=None) if timestamp.tzinfo else timestamp
    
    elif isinstance(timestamp, str):
        # ISO format string - handle with/without 'Z' suffix
        logger.debug(f"Parsing ISO format string: {timestamp}")
        try:
            # Replace 'Z' with '+00:00' for proper ISO parsing
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            # Strip timezone info to return naive datetime
            return dt.replace(tzinfo=None)
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid ISO format string: {timestamp}") from e
    
    elif isinstance(timestamp, (int, float)):
        # Unix timestamp (seconds since epoch)
        logger.debug(f"Parsing Unix timestamp: {timestamp}")
        try:
            return datetime.fromtimestamp(timestamp)
        except (ValueError, OSError) as e:
            raise ValueError(f"Invalid Unix timestamp: {timestamp}") from e
    
    else:
        raise ValueError(
            f"Unsupported timestamp type: {type(timestamp)}. "
            f"Expected str, datetime, int, or float."
        )


def format_timestamp(dt: datetime, format_string: str = "%Y%m%d_%H%M%S") -> str:
    """
    Format datetime object to string using consistent format.
    
    Args:
        dt: datetime object to format
        format_string: strftime format string (default: "%Y%m%d_%H%M%S" for filenames)
    
    Returns:
        Formatted string
    
    Raises:
        ValueError: If dt is not a datetime object
        TypeError: If format_string is not a string
    
    Examples:
        >>> dt = datetime(2025, 10, 25, 18, 23, 19)
        >>> format_timestamp(dt)
        '20251025_182319'
        
        >>> format_timestamp(dt, "%Y-%m-%d %H:%M:%S")
        '2025-10-25 18:23:19'
    """
    if not isinstance(dt, datetime):
        raise ValueError(f"Expected datetime object, got {type(dt)}")
    
    if not isinstance(format_string, str):
        raise TypeError(f"format_string must be str, got {type(format_string)}")
    
    try:
        return dt.strftime(format_string)
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Invalid format string: {format_string}") from e
