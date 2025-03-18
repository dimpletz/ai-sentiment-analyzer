"""
Sentiment analyzer implementation.

This module provides the core sentiment analysis functionality using the DistilBERT model.
The SentimentAnalyzer class handles text preprocessing and prediction, utilizing
the ModelManager for model access and device management.

Author: Ofelia Webb <ofelia.b.webb@gmail.com>
"""

import torch
from typing import Dict, List, Union

from app.config.settings import (
    MAX_LENGTH,
    SENTIMENT_LABELS,
    CONFIDENCE_THRESHOLD
)
from app.models.model_manager import ModelManager

class SentimentAnalyzer:
    """
    A class for performing sentiment analysis on text using DistilBERT.
    
    This class handles the sentiment analysis process, utilizing the ModelManager
    for model access and device management. It supports both single text and batch
    processing with confidence scoring.
    
    Attributes:
        model_manager (ModelManager): The manager for model and device handling
    """
    
    def __init__(self):
        """
        Initialize the sentiment analyzer.
        
        This method creates a new instance of the ModelManager to handle
        model loading and device management.
        """
        self.model_manager = ModelManager()

    def analyze(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Analyze the sentiment of the input text(s).
        
        This method handles both single text and batch processing. For single text,
        it returns a dictionary with the analysis results. For batch processing,
        it returns a list of dictionaries, one for each input text.
        
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
            >>> analyzer = SentimentAnalyzer()
            >>> # Single text
            >>> result = analyzer.analyze("Great product!")
            >>> # Batch processing
            >>> results = analyzer.analyze(["Great!", "Terrible!", "Okay"])
        """
        if isinstance(text, str):
            return self._analyze_single(text)
        return self._analyze_batch(text)

    def _analyze_single(self, text: str) -> Dict:
        """
        Analyze sentiment for a single text.
        
        This internal method handles the actual sentiment analysis process for a single
        text input, including tokenization, model inference, and result formatting.
        
        Args:
            text: The text string to analyze
            
        Returns:
            A dictionary containing the analysis results with the following keys:
                - text: The original input text
                - sentiment: The predicted sentiment (POSITIVE/NEGATIVE)
                - confidence: The confidence score (0-1)
                - is_confident: Boolean indicating if confidence exceeds threshold
        """
        model, tokenizer = self.model_manager.get_model_and_tokenizer()
        device = self.model_manager.get_device()
        
        inputs = tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(scores, dim=1).item()
            confidence = scores[0][prediction].item()

        return {
            "text": text,
            "sentiment": SENTIMENT_LABELS[prediction],
            "confidence": confidence,
            "is_confident": confidence >= CONFIDENCE_THRESHOLD
        }

    def _analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment for a batch of texts.
        
        This internal method processes multiple texts by calling _analyze_single
        for each text in the input list.
        
        Args:
            texts: A list of text strings to analyze
            
        Returns:
            A list of dictionaries, where each dictionary contains the analysis
            results for the corresponding input text
        """
        return [self._analyze_single(text) for text in texts] 