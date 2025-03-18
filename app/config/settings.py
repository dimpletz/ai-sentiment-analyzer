"""
Configuration settings for the sentiment analyzer.
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