"""
Tests for the main module.
"""
import pytest
from app.main import main

def test_main():
    """
    Test the main function.
    """
    assert main() == 0