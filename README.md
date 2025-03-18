# AI Sentiment Analyzer

A powerful sentiment analysis tool built with Python using DistilBERT, designed to analyze text and determine sentiment with high confidence.

## Features

- **Accurate Sentiment Analysis**: Uses DistilBERT (distilbert-base-uncased-finetuned-sst-2-english) for state-of-the-art sentiment analysis
- **Confidence Scoring**: Provides confidence scores for each prediction
- **Batch Processing**: Analyze multiple texts simultaneously
- **Summary Statistics**: Generate comprehensive statistics about sentiment distribution
- **GPU Support**: Automatic GPU utilization when available for faster processing
- **Easy Integration**: Simple API for integration into other projects

## Requirements

- Python 3.8.1 or higher
- Poetry for dependency management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dimpletz/ai-sentiment-analyzer.git
cd ai-sentiment-analyzer
```

2. Install dependencies using Poetry:
```bash
poetry install
```

## Usage

### Running the Application

You can run the application in two ways:

1. Using Poetry:
```bash
poetry run python app/main.py
```

2. As a Python module:
```bash
poetry run python -m app.main
```

### Example Output

```
Individual Results:

Text: I absolutely love this product! It's amazing!
Sentiment: POSITIVE
Confidence: 1.00
Confident Prediction: True

Text: This is the worst experience ever.
Sentiment: NEGATIVE
Confidence: 1.00
Confident Prediction: True

Summary Statistics:
Total Texts: 2
Positive: 1 (50.0%)
Negative: 1 (50.0%)
Confident Predictions: 2 (100.0%)
```

### Using in Your Code

```python
from app.utils.sentiment_utils import analyze_sentiment, get_sentiment_summary

# Analyze a single text
result = analyze_sentiment("I love this product!")
print(result)
# Output: {'text': 'I love this product!', 'sentiment': 'POSITIVE', 'confidence': 0.98, 'is_confident': True}

# Analyze multiple texts
texts = [
    "I absolutely love this product!",
    "This is the worst experience ever.",
    "The weather is okay today."
]
results = analyze_sentiment(texts)

# Get summary statistics
summary = get_sentiment_summary(results)
```

## Project Structure

```
ai-sentiment-analyzer/
├── app/
│   ├── config/
│   │   └── settings.py         # Configuration settings
│   ├── models/
│   │   └── model_manager.py    # Model loading and management
│   ├── utils/
│   │   ├── sentiment_utils.py  # High-level utility functions
│   │   └── sentiment_analyzer.py# Core sentiment analysis logic
│   └── main.py                 # Application entry point
├── tests/
│   ├── unit/
│   │   ├── test_sentiment_analyzer.py  # Unit tests for analyzer
│   │   └── test_sentiment_utils.py     # Unit tests for utilities
│   ├── integration/
│   │   └── test_end_to_end.py         # Integration tests
│   └── conftest.py                     # Test fixtures
├── poetry.lock                 # Lock file for dependencies
├── pyproject.toml             # Project configuration
└── README.md                  # Project documentation
```

## Testing

### Running Tests

Run the test suite using Poetry:
```bash
poetry run pytest
```

Run tests with coverage report:
```bash
poetry run pytest --cov=app tests/
```

### Test Categories

1. **Unit Tests**
   - Sentiment analysis accuracy
   - Confidence score calculation
   - Batch processing functionality
   - Model loading and management
   - Utility function behavior

2. **Integration Tests**
   - End-to-end text analysis workflow
   - Model loading and prediction pipeline
   - Batch processing with various text types
   - Error handling and edge cases

3. **Performance Tests**
   - Processing speed benchmarks
   - Memory usage monitoring
   - GPU utilization efficiency
   - Batch processing optimization

### Test Coverage Goals

- Minimum 85% code coverage
- 100% coverage for core sentiment analysis logic
- All public APIs must be tested
- Edge cases and error conditions covered

### Development Workflow

1. Write tests before implementing features (TDD)
2. Run tests locally before committing:
   ```bash
   # Format code
   poetry run black .
   poetry run isort .
   
   # Run type checking
   poetry run mypy .
   
   # Run tests
   poetry run pytest
   ```
3. Ensure all tests pass before submitting PR
4. Include new tests for bug fixes

## Configuration

The sentiment analyzer can be configured through `app/config/settings.py`:

- `MODEL_NAME`: The HuggingFace model to use
- `MAX_LENGTH`: Maximum sequence length for tokenization
- `BATCH_SIZE`: Batch size for processing
- `CONFIDENCE_THRESHOLD`: Threshold for confident predictions

## Technical Details

### Model

The analyzer uses DistilBERT, a lightweight version of BERT that maintains good performance while being faster and requiring less computational resources. Specifically, it uses the `distilbert-base-uncased-finetuned-sst-2-english` model, which is fine-tuned for sentiment analysis on the SST-2 (Stanford Sentiment Treebank) dataset.

### Sentiment Classification

- **POSITIVE**: Indicates positive sentiment (confidence score > 0.5)
- **NEGATIVE**: Indicates negative sentiment (confidence score ≤ 0.5)
- **Confidence Score**: Range from 0 to 1, indicating the model's certainty
- **Confident Prediction**: Boolean flag indicating if the confidence exceeds the threshold

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- HuggingFace Transformers library
- DistilBERT model creators
- PyTorch team

## Contact

Project Link: https://github.com/dimpletz/ai-sentiment-analyzer