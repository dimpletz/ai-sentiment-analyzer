"""
Tests for the utils module.
"""
import pytest
from app.utils import example_util_function

def test_example_util_function():
    """
    Test the example utility function.
    """
    assert example_util_function() == "This is an example utility function"