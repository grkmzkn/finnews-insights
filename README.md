# Financial News Insights

This project implements advanced Natural Language Processing (NLP) techniques for financial news analysis, including Named Entity Recognition (NER) and Sentiment Analysis.

## Project Overview

The project aims to provide comprehensive analysis of financial news through multiple NLP techniques:

1. Named Entity Recognition (NER):
   - Companies
   - People (CEOs, executives, etc.)
   - Locations
   - Organizations
   - Dates
   - Financial metrics

2. Sentiment Analysis:
   - Multi-class sentiment classification (Positive, Negative, Neutral)
   - Entity-specific sentiment analysis
   - Advanced text preprocessing for financial context
   - Multiple model comparisons and optimizations

## Project Structure

```
finnews-insights/
├── data/                      # Data directory
│   ├── sentiment_data.csv     # Dataset for sentiment analysis
│   ├── ner_train_data.xlsx   # NER training data
│   ├── ner_valid_data.xlsx   # NER validation data
│   ├── ner_test_data.xlsx    # NER test data
│   └── label.json            # Label configurations
├── src/                      # Source code
│   ├── sentiment_analysis.ipynb  # Sentiment analysis implementation
│   ├── model_spacy.ipynb     # SpaCy NER implementation
│   └── model_bert.ipynb      # BERT implementations
└── results/                  # Model outputs and evaluation results
```

## Features

### Sentiment Analysis
- Multi-model comparison including:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - Naive Bayes
  - XGBoost
  - LightGBM
- Advanced text preprocessing pipeline:
  - URL and email removal
  - Stock symbol handling
  - Financial-specific stop words
  - Lemmatization
- TF-IDF vectorization with optimized parameters
- Model performance comparison and visualization
- Hyperparameter optimization using GridSearchCV

### Named Entity Recognition (NER)
- Implementation of two different NER approaches:
  - SpaCy custom NER model
  - BERT-based NER model
- Specialized entity recognition for financial context
- Model training with customizable parameters
- Comprehensive evaluation metrics
- Easy-to-use inference pipeline

### Planned Integration
- Entity-specific sentiment analysis
- Historical trend analysis
- Real-time news processing
- Interactive visualization dashboard

## Requirements

- Python 3.10+
- SpaCy
- PyTorch
- Transformers
- Pandas
- NumPy
- tqdm

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/grkmzkn/finnews-insights.git
cd finnews-insights
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preparation:
   - Place your training, validation, and test data in the `data` directory
   - Ensure data is in the correct format (see data format section)

2. Training:
   - Open and run the respective notebook for your chosen model:
     - `src/model_spacy.ipynb` for SpaCy model
     - `src/model_bert.ipynb` for BERT model

3. Inference:
   - Load the trained model
   - Use the provided test functions to make predictions on new text

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