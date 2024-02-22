"""
=================
Custom Exceptions
=================

Contains custom-made exceptions required to handle
errors in a way fine-tuned to WBE applications.
"""


class FunctionNotImplementedError(Exception):
    """
    This class handles the error when a function has not been implemented yet.
    Used in the development process of pyWBE.
    """
    def __init__(self, message: str) -> None:
        super().__init__(message)


class DurationExceededError(Exception):
    """
    This class handles the error when window duration is greater than the
    data duration.
    """
    def __init__(self, message: str) -> None:
        super().__init__(message)


class DurationTooShortError(Exception):
    """
    This class handles the error when window duration is too small for
    training.
    """
    def __init__(self, message: str) -> None:
        super().__init__(message)
