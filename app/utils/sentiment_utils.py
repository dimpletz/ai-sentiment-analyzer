"""
Utility functions for sentiment analysis.

This module provides high-level utility functions for sentiment analysis,
including a singleton pattern for the SentimentAnalyzer instance and
functions for analyzing text and generating summary statistics.

Author: Ofelia Webb <ofelia.b.webb@gmail.com>

Example:
    >>> # Analyze a single text
    >>> result = analyze_sentiment("I love this product!")
    >>> print(result)
    {'text': 'I love this product!', 'sentiment': 'POSITIVE', 'confidence': 0.98, 'is_confident': True}
    
    >>> # Analyze multiple texts and get summary
    >>> results = analyze_sentiment(["Great!", "Terrible!", "Okay"])
    >>> summary = get_sentiment_summary(results)
    >>> print(summary)
    {'total_texts': 3, 'positive_count': 1, 'negative_count': 1, ...}
"""

from typing import Dict, List, Union
from app.utils.sentiment_analyzer import SentimentAnalyzer

# Initialize the sentiment analyzer as a singleton
_analyzer = None

def get_analyzer() -> SentimentAnalyzer:
    """
    Get or create the sentiment analyzer instance.
    
    This function implements a singleton pattern for the SentimentAnalyzer,
    ensuring only one instance is created and reused throughout the application.
    
    Returns:
        SentimentAnalyzer: The singleton instance of the sentiment analyzer
    """
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer

def analyze_sentiment(text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
    """
    Analyze the sentiment of the input text(s).
    
    This is a convenience function that uses the singleton SentimentAnalyzer
    instance to perform sentiment analysis on one or more texts.
    
    Args:
        text: Either a single text string or a list of text strings to analyze
        
    Returns:
        For single text: A dictionary containing:
            - text: The original input text
            - sentiment: The predicted sentiment (POSITIVE/NEGATIVE)
            - confidence: The confidence score (0-1)
            - is_confident: Boolean indicating if confidence exceeds threshold
        For batch: A list of such dictionaries
        
    Example:
        >>> # Single text
        >>> result = analyze_sentiment("Great product!")
        >>> # Batch processing
        >>> results = analyze_sentiment(["Great!", "Terrible!", "Okay"])
    """
    analyzer = get_analyzer()
    return analyzer.analyze(text)

def get_sentiment_summary(results: List[Dict]) -> Dict:
    """
    Generate a summary of sentiment analysis results.
    
    This function processes a list of sentiment analysis results and generates
    summary statistics, including counts and percentages of positive and negative
    sentiments, as well as confidence metrics.
    
    Args:
        results: A list of sentiment analysis result dictionaries
        
    Returns:
        A dictionary containing summary statistics with the following keys:
            - total_texts: Total number of texts analyzed
            - positive_count: Number of positive sentiments
            - negative_count: Number of negative sentiments
            - confident_predictions: Number of confident predictions
            - positive_percentage: Percentage of positive sentiments
            - negative_percentage: Percentage of negative sentiments
            - confidence_rate: Percentage of confident predictions
            
    Example:
        >>> results = analyze_sentiment(["Great!", "Terrible!", "Okay"])
        >>> summary = get_sentiment_summary(results)
        >>> print(summary)
        {'total_texts': 3, 'positive_count': 1, 'negative_count': 1, ...}
    """
    total = len(results)
    positive = sum(1 for r in results if r["sentiment"] == "POSITIVE")
    negative = sum(1 for r in results if r["sentiment"] == "NEGATIVE")
    confident = sum(1 for r in results if r["is_confident"])
    
    return {
        "total_texts": total,
        "positive_count": positive,
        "negative_count": negative,
        "confident_predictions": confident,
        "positive_percentage": (positive / total) * 100 if total > 0 else 0,
        "negative_percentage": (negative / total) * 100 if total > 0 else 0,
        "confidence_rate": (confident / total) * 100 if total > 0 else 0
    } 