"""
Main application module.

This module serves as the entry point for the sentiment analyzer application.
It demonstrates the usage of the sentiment analysis functionality with example texts
and displays both individual results and summary statistics.

Author: Ofelia Webb <ofelia.b.webb@gmail.com>

Example:
    $ python -m app.main
    Individual Results:
    Text: I absolutely love this product! It's amazing!
    Sentiment: POSITIVE
    Confidence: 0.98
    Confident Prediction: True
    ...
"""

import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.utils.sentiment_utils import analyze_sentiment, get_sentiment_summary

def main() -> int:
    """
    Main function of the application.
    
    This function demonstrates the usage of the sentiment analyzer by:
    1. Analyzing a set of example texts
    2. Displaying individual results for each text
    3. Generating and displaying summary statistics
    
    Returns:
        int: Exit code (0 for success)
    """
    # Example texts to analyze
    texts = [
        "I absolutely love this product! It's amazing!",
        "This is the worst experience ever.",
        "The weather is okay today.",
        "The service was excellent and the staff was very friendly.",
        "I'm not sure how I feel about this.",
        "I love dogs",
        "I don't know what to think",
        "I love chocolate but I hate chocolate cake"
    ]
    
    # Analyze sentiments
    results = analyze_sentiment(texts)
    
    # Print individual results
    print("\nIndividual Results:")
    for result in results:
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Confident Prediction: {result['is_confident']}")
    
    # Print summary
    summary = get_sentiment_summary(results)
    print("\nSummary Statistics:")
    print(f"Total Texts: {summary['total_texts']}")
    print(f"Positive: {summary['positive_count']} ({summary['positive_percentage']:.1f}%)")
    print(f"Negative: {summary['negative_count']} ({summary['negative_percentage']:.1f}%)")
    print(f"Confident Predictions: {summary['confident_predictions']} ({summary['confidence_rate']:.1f}%)")
    
    return 0

if __name__ == "__main__":
    exit(main())