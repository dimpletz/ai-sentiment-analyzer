"""
Model manager for handling model loading and device management.

This module provides a centralized way to manage model loading, device selection,
and model initialization across the application.

Author: Ofelia Webb <ofelia.b.webb@gmail.com>
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Tuple

from app.config.settings import MODEL_NAME

class ModelManager:
    """
    A class for managing model loading and device selection.
    
    This class handles the initialization of models and tokenizers, ensuring
    proper device placement (CPU/GPU) and providing a consistent interface for
    model access across the application.
    
    Attributes:
        device (torch.device): The device (CPU/GPU) the model is running on
        tokenizer (AutoTokenizer): The tokenizer for text preprocessing
        model (AutoModelForSequenceClassification): The loaded sentiment analysis model
    """
    
    def __init__(self):
        """
        Initialize the model manager.
        
        This method:
        1. Determines the appropriate device (CPU/GPU)
        2. Loads the tokenizer and model
        3. Moves the model to the appropriate device
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
    
    def get_model_and_tokenizer(self) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """
        Get the loaded model and tokenizer.
        
        Returns:
            Tuple containing:
                - The loaded model
                - The tokenizer
        """
        return self.model, self.tokenizer
    
    def get_device(self) -> torch.device:
        """
        Get the current device being used.
        
        Returns:
            The torch device being used (CPU/GPU)
        """
        return self.device 