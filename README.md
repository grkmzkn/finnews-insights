# Financial News Insights

Advanced Natural Language Processing (NLP) project for financial news analysis, featuring Named Entity Recognition (NER) and Sentiment Analysis capabilities.

## Project Overview

This project implements sophisticated NLP techniques to analyze financial news content through:

1. Named Entity Recognition (NER):
   - Implements both SpaCy and BERT-based models
   - Detects and classifies entities such as:
     - Organizations (ORG)
     - People (PERSON)
     - Dates (DATE)
     - Monetary Values (MONEY)
     - Percentages (PERCENT)

2. Sentiment Analysis:
   - Multi-class sentiment classification
   - Categories: Positive, Negative, Neutral
   - Includes advanced text preprocessing for financial context

## Project Structure

```
finnews-insights/
├── data/                        # Data files and configurations
│   ├── sentiment_data.csv      # Sentiment analysis dataset
│   ├── label.json             # Entity label configurations
│   ├── train*.json            # Training data files
│   ├── valid.json            # Validation dataset
│   ├── test.json             # Test dataset
│   └── results/              # Training results and metrics
├── models/                     # Trained model files
│   ├── bert_model/           # BERT NER model files
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── tokenizer files
│   └── spacy_model/          # SpaCy NER model files
│       ├── config.cfg
│       └── model files
├── src/                        # Source code
│   ├── data_preparation.ipynb # Data preprocessing
│   ├── main.py               # Main application script
│   ├── model_bert.ipynb      # BERT model training
│   ├── model_spacy.ipynb     # SpaCy model training
│   └── sentiment_analysis.ipynb # Sentiment analysis
└── results/                    # Output and evaluation results
```

## Features

### Named Entity Recognition (NER)
- Dual model implementation:
  1. SpaCy Custom NER Model
     - Lightweight and efficient
     - Custom-trained for financial entities
     - Fast inference capabilities
  
  2. BERT-based NER Model
     - High accuracy for complex contexts
     - Transformer architecture
     - Better handling of ambiguous cases

### Sentiment Analysis
- Multi-class sentiment classification
- Advanced preprocessing pipeline:
  - URL and email removal
  - Stock symbol handling ($AAPL, $GOOG etc.)
  - Financial-specific preprocessing
  - Lemmatization and stop word removal
- TF-IDF vectorization
- ML model implementations

## Requirements

Main dependencies:
- Python 3.10+
- SpaCy 3.8.7
- PyTorch 2.9.0
- Transformers 4.57.1
- Pandas 2.3.3
- NumPy 2.2.6
- scikit-learn 1.7.2
- NLTK
- tqdm

Full requirements are available in `requirements.txt`.

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/grkmzkn/finnews-insights.git
cd finnews-insights
```

2. Create and activate a virtual environment:
```bash
python -m venv env
# On Windows:
env\Scripts\activate
# On Unix/MacOS:
source env/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation
- Run `data_preparation.ipynb` to process and prepare your data
- Ensure your data follows the required format (see data files for examples)

### 2. Model Training

#### NER Models:
1. SpaCy NER:
   - Open `src/model_spacy.ipynb`
   - Follow the notebook cells for training
   - Model will be saved in `models/spacy_model/`

2. BERT NER:
   - Open `src/model_bert.ipynb`
   - Execute training pipeline
   - Model will be saved in `models/bert_model/`

#### Sentiment Analysis:
- Use `src/sentiment_analysis.ipynb` for training sentiment analysis model

### 3. Inference
Use `main.py` for making predictions with trained models:
- NER detection using both SpaCy and BERT models
- Sentiment analysis on text
- Combined analysis capabilities

Example usage:
```python
# The main.py script provides integrated functionality
text = "Apple CEO Tim Cook announced a new project that boosted investor confidence."
# Will provide both NER and sentiment analysis results
```

## Model Performance

The current model achieves the following performance metrics on the validation set:
- Precision: [Score]
- Recall: [Score]
- F1 Score: [Score]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- GitHub: [@grkmzkn](https://github.com/grkmzkn)