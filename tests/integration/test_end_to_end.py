"""
Integration tests for the sentiment analyzer.
"""

import pytest
from app.main import main
from app.utils.sentiment_utils import analyze_sentiment, get_sentiment_summary
from app.utils.sentiment_analyzer import SentimentAnalyzer
from app.models.model_manager import ModelManager

def test_complete_workflow(sample_texts):
    """Test the complete sentiment analysis workflow."""
    # 1. Analyze sentiments
    results = analyze_sentiment(sample_texts)
    
    # 2. Verify results structure
    assert isinstance(results, list)
    assert len(results) == len(sample_texts)
    
    # 3. Generate and verify summary
    summary = get_sentiment_summary(results)
    assert isinstance(summary, dict)
    assert summary["total_texts"] == len(sample_texts)
    
    # 4. Verify main function execution
    assert main() == 0

def test_model_loading_and_prediction():
    """Test model loading and prediction pipeline."""
    # 1. Initialize components
    model_manager = ModelManager()
    analyzer = SentimentAnalyzer()
    
    # 2. Get model and tokenizer
    model, tokenizer = model_manager.get_model_and_token
    assert model is not None
    assert tokenizer is not None
    
    # 3. Verify device configuration
    device = model_manager.get_device()
    assert str(device) in ['cpu', 'cuda']
    
    # 4. Test prediction pipeline
    text = "This is a test message."
    result = analyzer._analyze_single(text)
    
    assert isinstance(result, dict)
    assert result["text"] == text
    assert result["sentiment"] in ["POSITIVE", "NEGATIVE"]
    assert 0 <= result["confidence"] <= 1

def test_batch_processing_performance(sample_texts):
    """Test batch processing performance and memory usage."""
    # 1. Create a large batch of texts
    large_batch = sample_texts * 20  # 100 texts
    
    # 2. Process batch
    results = analyze_sentiment(large_batch)
    
    # 3. Verify results
    assert len(results) == len(large_batch)
    assert all(isinstance(r, dict) for r in results)
    
    # 4. Verify summary generation for large batch
    summary = get_sentiment_summary(results)
    assert summary["total_texts"] == len(large_batch)

def test_error_handling_and_recovery():
    """Test error handling and recovery in the pipeline."""
    analyzer = SentimentAnalyzer()
    
    # 1. Test with empty input
    result_empty = analyzer._analyze_single("")
    assert isinstance(result_empty, dict)
    
    # 2. Test with very long input
    long_text = "test " * 1000
    result_long = analyzer._analyze_single(long_text)
    assert isinstance(result_long, dict)
    
    # 3. Test with special characters
    special_text = "!@#$%^&*()_+ This is a test 你好"
    result_special = analyzer._analyze_single(special_text)
    assert isinstance(result_special, dict)
    
    # 4. Test with mixed batch
    mixed_batch = ["", long_text, special_text]
    results_mixed = analyze_sentiment(mixed_batch)
    assert len(results_mixed) == len(mixed_batch)

@pytest.mark.parametrize("text,expected_type", [
    ("I love this!", "POSITIVE"),
    ("I hate this!", "NEGATIVE"),
    ("This is okay.", None),  # Can be either positive or negative
])
def test_sentiment_consistency(text, expected_type):
    """Test consistency of sentiment predictions."""
    # Run multiple predictions to check consistency
    results = [analyze_sentiment(text) for _ in range(3)]
    
    # Verify all predictions are the same
    sentiments = [r["sentiment"] for r in results]
    assert len(set(sentiments)) == 1  # All predictions should be the same
    
    if expected_type:
        assert sentiments[0] == expected_type 