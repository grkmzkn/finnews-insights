# Financial News Insights - NER Project

This project implements Named Entity Recognition (NER) for financial news using both SpaCy and BERT models.

## Project Overview

The project aims to extract named entities from financial news articles, focusing on entities such as:
- Companies
- People (CEOs, executives, etc.)
- Locations
- Organizations
- Dates
- Financial metrics

## Project Structure

```
finnews-insights/
├── data/                    # Data directory
│   ├── final_train_data.xlsx
│   ├── final_valid_data.xlsx
│   ├── final_test_data.xlsx
│   └── label.json
├── src/                    # Source code
│   ├── model_spacy.ipynb   # SpaCy implementation
│   └── model_bert.ipynb    # BERT implementation
└── results/                # Model outputs and evaluation results
```

## Features

- Data preprocessing and preparation
- Implementation of two different NER approaches:
  - SpaCy custom NER model
  - BERT-based NER model
- Model training with customizable parameters
- Model evaluation and performance metrics
- Easy-to-use inference pipeline for new texts

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