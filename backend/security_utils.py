"""
Security Utilities Module
Path validation and security utilities for PCA application.

This module provides secure file and path handling functions to prevent
directory traversal attacks and ensure safe file operations.
"""

import os
import pathlib
from typing import Optional, Union, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


def validate_file_path(file_path: Union[str, Path], allowed_extensions: Optional[List[str]] = None,
                      base_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Validate and secure a file path to prevent directory traversal attacks.

    Args:
        file_path: The file path to validate
        allowed_extensions: List of allowed file extensions (e.g., ['.xlsx', '.csv'])
        base_dir: Base directory to restrict paths to (optional)

    Returns:
        Path: Resolved and validated Path object

    Raises:
        SecurityError: If path validation fails
        FileNotFoundError: If file doesn't exist
    """
    try:
        # Convert to Path object
        path = Path(file_path).resolve()

        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.is_file():
            raise SecurityError(f"Path is not a file: {file_path}")

        # Check file extension if specified
        if allowed_extensions:
            if not any(path.suffix.lower() == ext.lower() for ext in allowed_extensions):
                raise SecurityError(f"File extension not allowed. Allowed: {allowed_extensions}")

        # Check if path is within base directory (if specified)
        if base_dir:
            base_path = Path(base_dir).resolve()
            try:
                path.relative_to(base_path)
            except ValueError:
                raise SecurityError(f"File path is outside allowed directory: {base_dir}")

        # Additional security checks
        _check_path_security(path)

        return path

    except Exception as e:
        logger.error(f"Path validation failed for {file_path}: {e}")
        if isinstance(e, (SecurityError, FileNotFoundError)):
            raise
        raise SecurityError(f"Invalid file path: {file_path}")


def validate_directory_path(dir_path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None,
                           create_if_missing: bool = False) -> Path:
    """
    Validate and secure a directory path.

    Args:
        dir_path: The directory path to validate
        base_dir: Base directory to restrict paths to (optional)
        create_if_missing: Whether to create directory if it doesn't exist

    Returns:
        Path: Resolved and validated Path object

    Raises:
        SecurityError: If path validation fails
    """
    try:
        path = Path(dir_path).resolve()

        # Check if directory exists or create it
        if not path.exists():
            if create_if_missing:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path}")
            else:
                raise FileNotFoundError(f"Directory not found: {dir_path}")

        if not path.is_dir():
            raise SecurityError(f"Path is not a directory: {dir_path}")

        # Check if path is within base directory (if specified)
        if base_dir:
            base_path = Path(base_dir).resolve()
            try:
                path.relative_to(base_path)
            except ValueError:
                raise SecurityError(f"Directory path is outside allowed directory: {base_dir}")

        # Additional security checks
        _check_path_security(path)

        return path

    except Exception as e:
        logger.error(f"Directory validation failed for {dir_path}: {e}")
        if isinstance(e, (SecurityError, FileNotFoundError)):
            raise
        raise SecurityError(f"Invalid directory path: {dir_path}")


def _check_path_security(path: Path) -> None:
    """
    Perform additional security checks on a path.

    Args:
        path: Path to check

    Raises:
        SecurityError: If security check fails
    """
    # Check for suspicious patterns
    path_str = str(path)

    # Prevent directory traversal attempts
    if '..' in path_str:
        # Allow '..' only if it's a legitimate parent directory reference
        # But be very strict about it
        parts = path.parts
        if any(part == '..' for part in parts):
            raise SecurityError("Directory traversal detected")

    # Check for null bytes (common in some attacks)
    if '\x00' in path_str:
        raise SecurityError("Null byte detected in path")

    # Check for very long paths that might cause issues
    if len(path_str) > 4096:  # Common filesystem limit
        raise SecurityError("Path too long")

    # Check for suspicious characters
    suspicious_chars = ['<', '>', '|', '*', '?']
    if any(char in path_str for char in suspicious_chars):
        raise SecurityError(f"Suspicious characters detected in path: {suspicious_chars}")


def secure_temp_file(suffix: str = '', prefix: str = 'pca_', dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Create a secure temporary file with proper permissions.

    Args:
        suffix: File suffix/extension
        prefix: File prefix
        dir: Directory to create file in (optional)

    Returns:
        Path: Path to the temporary file

    Raises:
        SecurityError: If temp file creation fails
    """
    import tempfile

    try:
        # Use system temp directory if no dir specified
        temp_dir = dir or tempfile.gettempdir()

        # Validate temp directory
        temp_path = validate_directory_path(temp_dir, create_if_missing=True)

        # Create temporary file
        fd, temp_file_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=str(temp_path))

        # Close the file descriptor (we'll reopen it when needed)
        os.close(fd)

        path = Path(temp_file_path)

        # Set restrictive permissions (readable/writable by owner only)
        if os.name == 'posix':  # Unix-like systems
            path.chmod(0o600)
        # Windows permissions are handled differently and are generally more restrictive by default

        logger.info(f"Created secure temp file: {path}")
        return path

    except Exception as e:
        logger.error(f"Failed to create secure temp file: {e}")
        raise SecurityError(f"Could not create secure temporary file: {e}")


def secure_temp_directory(suffix: str = '', prefix: str = 'pca_', dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Create a secure temporary directory.

    Args:
        suffix: Directory suffix
        prefix: Directory prefix
        dir: Parent directory (optional)

    Returns:
        Path: Path to the temporary directory

    Raises:
        SecurityError: If temp directory creation fails
    """
    import tempfile

    try:
        temp_dir = dir or tempfile.gettempdir()
        temp_path = validate_directory_path(temp_dir, create_if_missing=True)

        temp_dir_path = Path(tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=str(temp_path)))

        # Set restrictive permissions
        if os.name == 'posix':
            temp_dir_path.chmod(0o700)

        logger.info(f"Created secure temp directory: {temp_dir_path}")
        return temp_dir_path

    except Exception as e:
        logger.error(f"Failed to create secure temp directory: {e}")
        raise SecurityError(f"Could not create secure temporary directory: {e}")


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize a filename to prevent security issues.

    Args:
        filename: Original filename
        max_length: Maximum allowed length

    Returns:
        str: Sanitized filename
    """
    import re

    # Remove or replace dangerous characters
    # Allow alphanumeric, dots, hyphens, underscores, and spaces
    sanitized = re.sub(r'[^\w\.-]', '_', filename)

    # Remove multiple consecutive dots (potential directory traversal)
    sanitized = re.sub(r'\.\.+', '.', sanitized)

    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip(' .')

    # Limit length
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:max_length-len(ext)] + ext

    # Ensure it's not empty and doesn't start with a dot (hidden files)
    if not sanitized or sanitized.startswith('.'):
        sanitized = f"file_{hash(filename) % 10000}"

    return sanitized


def get_secure_base_dir() -> Path:
    """
    Get the secure base directory for the application.

    Returns:
        Path: Secure base directory path
    """
    # Use current working directory as base, but ensure it's secure
    base_dir = Path.cwd()

    # Additional validation for base directory
    if not base_dir.exists() or not base_dir.is_dir():
        raise SecurityError("Invalid base directory")

    return base_dir


def validate_url(url: str) -> bool:
    """
    Basic URL validation to prevent malicious URLs.

    Args:
        url: URL to validate

    Returns:
        bool: True if URL appears safe
    """
    import re

    # Basic URL pattern
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    if not url_pattern.match(url):
        return False

    # Additional checks
    if '..' in url or '<' in url or '>' in url:
        return False

    return True