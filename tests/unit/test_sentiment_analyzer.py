"""
Unit tests for the sentiment analyzer.

This module contains unit tests for the core sentiment analyzer functionality,
including initialization, text processing, and sentiment prediction.

Author: Ofelia Webb <ofelia.b.webb@gmail.com>
"""

import pytest
from app.utils.sentiment_analyzer import SentimentAnalyzer
from app.config.settings import CONFIDENCE_THRESHOLD

def test_sentiment_analyzer_initialization(sentiment_analyzer):
    """Test sentiment analyzer initialization."""
    assert isinstance(sentiment_analyzer, SentimentAnalyzer)
    assert sentiment_analyzer.model_manager is not None

def test_analyze_single_positive(sentiment_analyzer, positive_text):
    """Test sentiment analysis for a single positive text."""
    result = sentiment_analyzer._analyze_single(positive_text)
    
    assert isinstance(result, dict)
    assert result["text"] == positive_text
    assert result["sentiment"] == "POSITIVE"
    assert result["confidence"] > CONFIDENCE_THRESHOLD
    assert result["is_confident"] is True

def test_analyze_single_negative(sentiment_analyzer, negative_text):
    """Test sentiment analysis for a single negative text."""
    result = sentiment_analyzer._analyze_single(negative_text)
    
    assert isinstance(result, dict)
    assert result["text"] == negative_text
    assert result["sentiment"] == "NEGATIVE"
    assert result["confidence"] > CONFIDENCE_THRESHOLD
    assert result["is_confident"] is True

def test_analyze_batch(sentiment_analyzer, sample_texts):
    """Test batch sentiment analysis."""
    results = sentiment_analyzer._analyze_batch(sample_texts)
    
    assert isinstance(results, list)
    assert len(results) == len(sample_texts)
    for result, text in zip(results, sample_texts):
        assert isinstance(result, dict)
        assert result["text"] == text
        assert result["sentiment"] in ["POSITIVE", "NEGATIVE"]
        assert 0 <= result["confidence"] <= 1
        assert isinstance(result["is_confident"], bool)

def test_analyze_empty_text(sentiment_analyzer):
    """Test handling of empty text."""
    result = sentiment_analyzer._analyze_single("")
    
    assert isinstance(result, dict)
    assert result["text"] == ""
    assert "sentiment" in result
    assert "confidence" in result
    assert "is_confident" in result

def test_analyze_long_text(sentiment_analyzer):
    """Test handling of long text."""
    long_text = "great " * 1000  # Create a very long text
    result = sentiment_analyzer._analyze_single(long_text)
    
    assert isinstance(result, dict)
    assert result["text"] == long_text
    assert "sentiment" in result
    assert "confidence" in result
    assert "is_confident" in result

@pytest.mark.parametrize("text,expected_sentiment", [
    ("I love this!", "POSITIVE"),
    ("This is terrible!", "NEGATIVE"),
    ("Amazing product!", "POSITIVE"),
    ("Worst experience ever!", "NEGATIVE"),
])
def test_various_sentiments(sentiment_analyzer, text, expected_sentiment):
    """Test sentiment analysis with various input texts."""
    result = sentiment_analyzer._analyze_single(text)
    assert result["sentiment"] == expected_sentiment

def test_confidence_threshold(sentiment_analyzer):
    """Test that confidence scores respect the threshold."""
    text = "This is a test."
    result = sentiment_analyzer._analyze_single(text)
    
    assert 0 <= result["confidence"] <= 1
    assert result["is_confident"] == (result["confidence"] >= CONFIDENCE_THRESHOLD) 