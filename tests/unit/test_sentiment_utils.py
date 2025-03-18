"""
Unit tests for sentiment utility functions.
"""

import pytest
from app.utils.sentiment_utils import analyze_sentiment, get_sentiment_summary

def test_analyze_sentiment_single(positive_text):
    """Test sentiment analysis for a single text."""
    result = analyze_sentiment(positive_text)
    
    assert isinstance(result, dict)
    assert result["text"] == positive_text
    assert result["sentiment"] in ["POSITIVE", "NEGATIVE"]
    assert 0 <= result["confidence"] <= 1
    assert isinstance(result["is_confident"], bool)

def test_analyze_sentiment_batch(sample_texts):
    """Test sentiment analysis for a batch of texts."""
    results = analyze_sentiment(sample_texts)
    
    assert isinstance(results, list)
    assert len(results) == len(sample_texts)
    for result, text in zip(results, sample_texts):
        assert isinstance(result, dict)
        assert result["text"] == text
        assert result["sentiment"] in ["POSITIVE", "NEGATIVE"]
        assert 0 <= result["confidence"] <= 1
        assert isinstance(result["is_confident"], bool)

def test_get_sentiment_summary_empty():
    """Test summary generation with empty results."""
    summary = get_sentiment_summary([])
    
    assert isinstance(summary, dict)
    assert summary["total_texts"] == 0
    assert summary["positive_count"] == 0
    assert summary["negative_count"] == 0
    assert summary["confident_predictions"] == 0
    assert summary["positive_percentage"] == 0
    assert summary["negative_percentage"] == 0
    assert summary["confidence_rate"] == 0

def test_get_sentiment_summary(sample_texts):
    """Test summary generation with sample texts."""
    results = analyze_sentiment(sample_texts)
    summary = get_sentiment_summary(results)
    
    assert isinstance(summary, dict)
    assert summary["total_texts"] == len(sample_texts)
    assert summary["positive_count"] + summary["negative_count"] == len(sample_texts)
    assert 0 <= summary["positive_percentage"] <= 100
    assert 0 <= summary["negative_percentage"] <= 100
    assert 0 <= summary["confidence_rate"] <= 100
    assert summary["confident_predictions"] <= len(sample_texts)

def test_analyze_sentiment_invalid_input():
    """Test handling of invalid input types."""
    with pytest.raises(TypeError):
        analyze_sentiment(123)
    
    with pytest.raises(TypeError):
        analyze_sentiment([123, 456])

def test_get_sentiment_summary_invalid_input():
    """Test handling of invalid input for summary generation."""
    with pytest.raises(TypeError):
        get_sentiment_summary("not a list")
    
    with pytest.raises(ValueError):
        get_sentiment_summary([{"invalid": "structure"}])

@pytest.mark.parametrize("results,expected_positive,expected_negative", [
    ([
        {"sentiment": "POSITIVE", "is_confident": True},
        {"sentiment": "NEGATIVE", "is_confident": True}
    ], 1, 1),
    ([
        {"sentiment": "POSITIVE", "is_confident": True},
        {"sentiment": "POSITIVE", "is_confident": False}
    ], 2, 0),
    ([
        {"sentiment": "NEGATIVE", "is_confident": True},
        {"sentiment": "NEGATIVE", "is_confident": True}
    ], 0, 2),
])
def test_get_sentiment_summary_counts(results, expected_positive, expected_negative):
    """Test summary statistics counting."""
    summary = get_sentiment_summary(results)
    assert summary["positive_count"] == expected_positive
    assert summary["negative_count"] == expected_negative 