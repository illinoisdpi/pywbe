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
