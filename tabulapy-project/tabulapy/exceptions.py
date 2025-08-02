# tabulapy/exceptions.py
class TabulaPyError(Exception):
    """Base exception for the TabulaPy library."""
    pass

class ConfigurationError(TabulaPyError):
    """For errors related to initial table setup."""
    pass

class ColumnNotFoundException(TabulaPyError, KeyError):
    """Raised when a specified column name or index is not found."""
    pass

class InvalidRuleException(TabulaPyError, ValueError):
    """Raised when an unsupported styling rule is requested."""
    pass

class InvalidStyleException(TabulaPyError, ValueError):
    """Raised when an unsupported styling format is requested."""
    pass