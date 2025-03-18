# AI Sentiment Analyzer

A powerful sentiment analysis tool using DistilBERT for accurate text sentiment classification. This tool provides fast and reliable sentiment analysis with confidence scoring and batch processing capabilities.

## Features

- **Advanced Sentiment Analysis**
  - Single text and batch text analysis
  - Binary sentiment classification (Positive/Negative)
  - Confidence scoring for each prediction
  - Configurable confidence threshold
  - Summary statistics for batch analysis

- **Technical Features**
  - GPU acceleration support (automatically uses CUDA if available)
  - Thread-safe implementation
  - Efficient batch processing
  - Memory-optimized model loading
  - Singleton pattern for resource management

- **Developer Features**
  - Type hints for better IDE support
  - Comprehensive documentation
  - Modular architecture
  - Easy to extend and customize

## Requirements

- Python 3.8.1 or higher
- CUDA-capable GPU (optional, for faster processing)
- Poetry for dependency management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-sentiment-analyzer.git
cd ai-sentiment-analyzer
```

2. Install dependencies using Poetry:
```bash
poetry install
```

## Usage

### Basic Usage

Run the example script to see the sentiment analyzer in action:
```bash
poetry run python -m app.main
```

### Using in Your Code

```python
from app.utils.sentiment_utils import analyze_sentiment, get_sentiment_summary

# Analyze a single text
result = analyze_sentiment("I love this product!")
print(result)
# Output: {
#     'text': 'I love this product!',
#     'sentiment': 'POSITIVE',
#     'confidence': 0.98,
#     'is_confident': True
# }

# Analyze multiple texts
texts = [
    "Great product!",
    "Terrible service",
    "It's okay",
    "The weather is nice today",
    "I'm not sure about this"
]
results = analyze_sentiment(texts)
print(results)

# Get summary statistics
summary = get_sentiment_summary(results)
print(summary)
# Output: {
#     'total_texts': 5,
#     'positive_count': 2,
#     'negative_count': 2,
#     'confident_predictions': 4,
#     'positive_percentage': 40.0,
#     'negative_percentage': 40.0,
#     'confidence_rate': 80.0
# }
```

### Advanced Usage

```python
from app.utils.sentiment_analyzer import SentimentAnalyzer

# Create a custom analyzer instance
analyzer = SentimentAnalyzer()

# Analyze with custom batch size
results = analyzer.analyze(["Text 1", "Text 2", "Text 3"])
```

## Project Structure

```
ai-sentiment-analyzer/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config/
│   │   └── settings.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_manager.py
│   └── utils/
│       ├── __init__.py
│       ├── sentiment_analyzer.py
│       └── sentiment_utils.py
├── pyproject.toml
└── README.md
```

## Configuration

You can modify the following settings in `app/config/settings.py`:
- `MODEL_NAME`: The pre-trained model to use
- `MAX_LENGTH`: Maximum sequence length for tokenization
- `CONFIDENCE_THRESHOLD`: Minimum confidence for predictions
- `SENTIMENT_LABELS`: Mapping of model outputs to sentiment labels

## Development

### Running Tests
```bash
poetry run pytest
```

### Code Formatting
```bash
poetry run black .
poetry run isort .
```

### Type Checking
```bash
poetry run mypy .
```

## Performance

- The model uses DistilBERT, a lighter and faster version of BERT
- GPU acceleration is automatically enabled when available
- Batch processing is optimized for memory efficiency
- The singleton pattern ensures efficient resource usage

## Author

Ofelia Webb <ofelia.b.webb@gmail.com>

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request