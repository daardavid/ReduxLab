"""
Secure Error Handler Module
Sanitized error handling for PCA application.

This module provides secure error handling that prevents information leakage
while maintaining useful debugging information for authorized users.
"""

import logging
import traceback
import sys
from typing import Optional, Dict, Any, Union
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class SanitizedError(Exception):
    """Base class for sanitized errors that don't leak sensitive information."""
    pass


class SecurityError(SanitizedError):
    """Security-related errors."""
    pass


class ValidationError(SanitizedError):
    """Data validation errors."""
    pass


class ProcessingError(SanitizedError):
    """Data processing errors."""
    pass


def sanitize_error_message(error: Exception, context: str = "") -> str:
    """
    Sanitize error messages to prevent information leakage.

    Args:
        error: The original exception
        context: Additional context about where the error occurred

    Returns:
        str: Sanitized error message safe for user display
    """
    error_type = type(error).__name__

    # Define patterns that indicate sensitive information
    sensitive_patterns = [
        r'password',
        r'key',
        r'token',
        r'secret',
        r'credential',
        r'auth',
        r'login',
        r'user.*name',
        r'email',
        r'path.*[/\\]',
        r'file.*[/\\]',
        r'directory.*[/\\]',
        r'C:\\',
        r'/home/',
        r'/usr/',
        r'/var/',
        r'\\Users\\',
        r'\\Program Files',
        r'\.env',
        r'config.*file',
        r'database.*url',
        r'connection.*string'
    ]

    # Get the original error message
    original_message = str(error)

    # Sanitize the message
    sanitized_message = original_message

    # Remove or mask sensitive information
    for pattern in sensitive_patterns:
        sanitized_message = re.sub(pattern, '[REDACTED]', sanitized_message, flags=re.IGNORECASE)

    # Limit message length to prevent very long error messages
    if len(sanitized_message) > 500:
        sanitized_message = sanitized_message[:497] + "..."

    # Create user-friendly message based on error type
    user_friendly_messages = {
        'FileNotFoundError': 'The specified file could not be found.',
        'PermissionError': 'Insufficient permissions to access the requested resource.',
        'ValueError': 'Invalid data or parameter provided.',
        'TypeError': 'Data type mismatch in operation.',
        'KeyError': 'Required data element is missing.',
        'IndexError': 'Data structure access error.',
        'AttributeError': 'Internal system error.',
        'ImportError': 'Required system component not available.',
        'OSError': 'System operation failed.',
        'MemoryError': 'Insufficient memory for operation.',
        'TimeoutError': 'Operation timed out.',
        'ConnectionError': 'Network or connection error.',
        'SecurityError': 'Security policy violation.',
        'ValidationError': 'Data validation failed.',
        'ProcessingError': 'Data processing failed.'
    }

    # Use user-friendly message if available, otherwise use sanitized version
    if error_type in user_friendly_messages:
        base_message = user_friendly_messages[error_type]
    else:
        base_message = f"An error occurred: {sanitized_message}"

    # Add context if provided
    if context:
        base_message = f"{context}: {base_message}"

    return base_message


def log_error_securely(error: Exception, context: str = "", user_visible: bool = False,
                       include_traceback: bool = True) -> str:
    """
    Log an error securely, separating user-visible messages from internal logs.

    Args:
        error: The exception that occurred
        context: Context about where the error occurred
        user_visible: Whether this error should be shown to users
        include_traceback: Whether to include traceback in logs

    Returns:
        str: User-safe error message
    """
    # Create sanitized user message
    user_message = sanitize_error_message(error, context)

    # Log detailed information for debugging (internal use only)
    logger.error(f"Error in {context}: {type(error).__name__}: {str(error)}")
    if include_traceback:
        logger.error(f"Traceback: {traceback.format_exc()}")

    # Log user-visible message if appropriate
    if user_visible:
        logger.info(f"User-visible error: {user_message}")

    return user_message


def handle_file_operation_error(error: Exception, file_path: Optional[str] = None,
                               operation: str = "file operation") -> str:
    """
    Handle file operation errors with appropriate sanitization.

    Args:
        error: The file operation exception
        file_path: Path to the file (will be sanitized)
        operation: Description of the operation

    Returns:
        str: Sanitized error message
    """
    context = f"File {operation}"

    # Sanitize file path if provided
    if file_path:
        try:
            from security_utils import sanitize_filename
            safe_filename = sanitize_filename(Path(file_path).name)
            context = f"{context} on '{safe_filename}'"
        except:
            context = f"{context} on file"

    return log_error_securely(error, context, user_visible=True)


def handle_data_processing_error(error: Exception, data_description: str = "data",
                                operation: str = "processing") -> str:
    """
    Handle data processing errors with appropriate sanitization.

    Args:
        error: The data processing exception
        data_description: Description of the data being processed
        operation: Description of the operation

    Returns:
        str: Sanitized error message
    """
    context = f"Data {operation}"

    # Sanitize data description
    if data_description:
        # Remove potential sensitive information from data description
        safe_description = re.sub(r'[^\w\s\-_\.]', '', data_description)
        safe_description = safe_description[:100]  # Limit length
        context = f"{context} for '{safe_description}'"

    return log_error_securely(error, context, user_visible=True)


def create_error_report(error: Exception, context: Dict[str, Any] = None,
                       include_system_info: bool = False) -> Dict[str, Any]:
    """
    Create a detailed error report for debugging while maintaining security.

    Args:
        error: The exception that occurred
        context: Additional context information (will be sanitized)
        include_system_info: Whether to include system information

    Returns:
        Dict[str, Any]: Error report dictionary
    """
    import platform
    import datetime

    report = {
        'timestamp': datetime.datetime.now().isoformat(),
        'error_type': type(error).__name__,
        'user_message': sanitize_error_message(error),
        'error_id': hex(hash(str(error) + str(datetime.datetime.now())))[2:10].upper()
    }

    # Add sanitized context if provided
    if context:
        sanitized_context = {}
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)):
                # Sanitize string values
                if isinstance(value, str):
                    sanitized_value = re.sub(r'[^\w\s\-_.,]', '', value)
                    sanitized_value = sanitized_value[:200]  # Limit length
                else:
                    sanitized_value = value
                sanitized_context[key] = sanitized_value
            else:
                # For complex objects, just store the type
                sanitized_context[key] = f"<{type(value).__name__}>"

        report['context'] = sanitized_context

    # Add basic system info if requested (no sensitive paths)
    if include_system_info:
        report['system_info'] = {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'architecture': platform.architecture()[0]
        }

    return report


def safe_exception_handler(func):
    """
    Decorator for safe exception handling in functions.

    This decorator catches exceptions, sanitizes them, and re-raises
    sanitized versions while logging the original error securely.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SanitizedError:
            # Re-raise sanitized errors as-is
            raise
        except Exception as e:
            # Sanitize and log other exceptions
            user_message = log_error_securely(e, f"Function {func.__name__}", user_visible=True)
            raise SanitizedError(user_message) from None

    return wrapper


class ErrorAggregator:
    """
    Aggregates multiple errors for batch reporting.

    Useful for operations that might generate multiple errors
    but should present a single, coherent message to the user.
    """

    def __init__(self):
        self.errors = []
        self.warnings = []

    def add_error(self, error: Exception, context: str = ""):
        """Add an error to the aggregator."""
        sanitized = sanitize_error_message(error, context)
        self.errors.append(sanitized)
        logger.error(f"Aggregated error: {sanitized}")

    def add_warning(self, message: str):
        """Add a warning to the aggregator."""
        self.warnings.append(message)
        logger.warning(f"Aggregated warning: {message}")

    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    def get_summary_message(self) -> str:
        """Get a summary message of all errors and warnings."""
        messages = []

        if self.errors:
            messages.append(f"{len(self.errors)} error(s) occurred")
            if len(self.errors) <= 3:
                messages.extend(f"• {error}" for error in self.errors)
            else:
                messages.extend(f"• {error}" for error in self.errors[:2])
                messages.append(f"• ... and {len(self.errors) - 2} more errors")

        if self.warnings:
            messages.append(f"{len(self.warnings)} warning(s)")
            if len(self.warnings) <= 3:
                messages.extend(f"• {warning}" for warning in self.warnings)
            else:
                messages.extend(f"• {warning}" for warning in self.warnings[:2])
                messages.append(f"• ... and {len(self.warnings) - 2} more warnings")

        return "\n".join(messages) if messages else "No errors or warnings"

    def raise_if_errors(self):
        """Raise an exception if there are errors."""
        if self.has_errors():
            raise SanitizedError(self.get_summary_message())


def validate_and_sanitize_input(input_data: Any, input_type: str = "general") -> Any:
    """
    Validate and sanitize input data based on type.

    Args:
        input_data: The input data to validate and sanitize
        input_type: Type of input ('filename', 'filepath', 'number', 'text', 'general')

    Returns:
        Any: Sanitized input data

    Raises:
        ValidationError: If input validation fails
    """
    if input_data is None:
        raise ValidationError("Input data cannot be None")

    if input_type == "filename":
        if not isinstance(input_data, str):
            raise ValidationError("Filename must be a string")

        from security_utils import sanitize_filename
        return sanitize_filename(input_data)

    elif input_type == "filepath":
        if not isinstance(input_data, (str, Path)):
            raise ValidationError("File path must be a string or Path object")

        from security_utils import validate_file_path
        try:
            return validate_file_path(input_data)
        except Exception as e:
            raise ValidationError(f"Invalid file path: {sanitize_error_message(e)}")

    elif input_type == "number":
        try:
            # Try to convert to float first, then to int if it's a whole number
            num = float(input_data)
            if num == int(num):
                return int(num)
            return num
        except (ValueError, TypeError):
            raise ValidationError("Invalid number format")

    elif input_type == "text":
        if not isinstance(input_data, str):
            raise ValidationError("Text input must be a string")

        # Basic text sanitization
        sanitized = re.sub(r'[^\w\s\-_.,!?()]', '', input_data)
        return sanitized.strip()

    elif input_type == "general":
        # For general input, just ensure it's not obviously malicious
        input_str = str(input_data)
        if len(input_str) > 10000:  # Reasonable limit
            raise ValidationError("Input data too large")

        # Check for obvious malicious patterns
        malicious_patterns = ['<script', 'javascript:', 'vbscript:', 'onload=', 'onerror=']
        for pattern in malicious_patterns:
            if pattern.lower() in input_str.lower():
                raise ValidationError("Potentially malicious input detected")

        return input_data

    else:
        raise ValidationError(f"Unknown input type: {input_type}")