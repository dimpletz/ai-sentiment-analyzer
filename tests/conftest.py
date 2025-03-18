"""
Common test fixtures and configurations.
"""

import pytest
from app.utils.sentiment_analyzer import SentimentAnalyzer
from app.models.model_manager import ModelManager

@pytest.fixture
def model_manager():
    """Fixture for ModelManager instance."""
    return ModelManager()

@pytest.fixture
def sentiment_analyzer():
    """Fixture for SentimentAnalyzer instance."""
    return SentimentAnalyzer()

@pytest.fixture
def sample_texts():
    """Fixture providing sample texts for testing."""
    return [
        "I absolutely love this product! It's amazing!",
        "This is the worst experience ever.",
        "The weather is okay today.",
        "The service was excellent and the staff was very friendly.",
        "I'm not sure how I feel about this."
    ]

@pytest.fixture
def positive_text():
    """Fixture for a positive text sample."""
    return "I absolutely love this product! It's amazing!"

@pytest.fixture
def negative_text():
    """Fixture for a negative text sample."""
    return "This is the worst experience ever."

@pytest.fixture
def neutral_text():
    """Fixture for a neutral/ambiguous text sample."""
    return "The weather is okay today." 