"""
Configuration settings for the sentiment analyzer.

This module contains all configuration parameters for the sentiment analyzer,
including model settings, thresholds, and labels.

Author: Ofelia Webb <ofelia.b.webb@gmail.com>
"""

# Model settings
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LENGTH = 512
BATCH_SIZE = 32

# Sentiment labels
SENTIMENT_LABELS = {
    0: "NEGATIVE",
    1: "POSITIVE"
}

# Confidence threshold for predictions
CONFIDENCE_THRESHOLD = 0.5 