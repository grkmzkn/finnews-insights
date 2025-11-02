import os
import pickle
import joblib
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
import torch

###########################################################################
# Load models
###########################################################################
# Load the TF-IDF vectorizer
with open(os.path.join("models", "tfidf_vector.pkl"), "rb") as file:
    tfidf = pickle.load(file)

# Load labels
with open('./data/label.json', 'r') as f:
    label_dict = json.load(f)

label2id = label_dict  # mappings from label.json
id2label = {v: k for k, v in label2id.items()}

# Load Sentiment model
sentiment_model = joblib.load("./models/sentiment_model.pkl")

# Load SpaCy model
spacy_model = spacy.load("./models/spacy_model")

# Load Bert model
bert_tokenizer = AutoTokenizer.from_pretrained("./models/bert_model")
bert_model = AutoModelForTokenClassification.from_pretrained("./models/bert_model")
############################################################################

###########################################################################
# Preprocess
###########################################################################
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove stock symbols (e.g., $AAPL, $GOOG)
    text = re.sub(r'\$\w+', '', text)
    
    # Remove numbers with % (percentage)
    text = re.sub(r'\d+%', '', text)
    
    # Remove currency symbols and amounts (e.g., $123.45, €100, £50)
    text = re.sub(r'[$€£¥]\d+(?:\.\d{2})?|\d+(?:\.\d{2})?[$€£¥]', '', text)
    
    # Remove special characters and numbers
    # Keep alphabets and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization with pos tagging for better accuracy
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]  # First try as verb
    tokens = [lemmatizer.lemmatize(token, pos='n') for token in tokens]  # Then as noun
    
    # Remove short words (length < 3)
    tokens = [token for token in tokens if len(token) > 2]
    
    return ' '.join(tokens)
###########################################################################

###########################################################################
# Sentiment Analysis
###########################################################################
def predict_sentiment(text):
    # Preprocess text
    processed = preprocess_text(text)
    # TF-IDF transformation and convert to dense array
    text_tfidf = tfidf.transform([processed]).toarray()
    
    # Get numeric prediction
    prediction_encoded = sentiment_model.predict(text_tfidf)[0]
    
    # Convert numeric prediction to sentiment label
    sentiment_labels = {0: 'neutral', 1: 'positive', 2: 'negative'}
    prediction = sentiment_labels[prediction_encoded]
    
    return prediction
###########################################################################

###########################################################################
# Bert Model Prediction
###########################################################################
def predict_entities_loaded(text, model, tokenizer, hardware='cpu'):
    """
    Identifies entities in the text using a loaded model.
    Args:
        text (str): Text to process
        model: Loaded model
        tokenizer: Loaded tokenizer
        hardware (str): Hardware to use - 'cpu' or 'gpu'
    """
    # Device selection
    device = 'cuda' if hardware.lower() == 'gpu' else 'cpu'

    # If GPU is selected but not available, warn and switch to CPU
    if device == 'cuda' and not torch.cuda.is_available():
        print("GPU not found, using CPU...")
        device = 'cpu'

    # Move model to selected device
    model.to(device)

    # Split text into words
    tokens = text.split()

    # Tokenize
    encoding = tokenizer(tokens, truncation=True, is_split_into_words=True, return_tensors="pt")
    word_ids = encoding.word_ids()

    # Move input tensors to selected device
    inputs = {k: v.to(device) for k, v in encoding.items()}

    # Predict
    with torch.no_grad():  # No gradient calculation
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(-1)

    # Convert predictions to labels
    predicted_labels = []
    for i, pred in enumerate(predictions[0]):
        if word_ids[i] is not None:  # if not a special token
            predicted_labels.append(id2label[pred.item()]) # Use the original id2label

    # Visualize results
    results = []
    for token, label in zip(tokens, predicted_labels):
        if label != 'O':
            results.append((token, label))

    return results
###########################################################################

###########################################################################
# Prediction
###########################################################################
# Test on a sample sentence
test_sentence = "Google CEO Sundar Pichai announced a new project."
doc = spacy_model(test_sentence)

print("\nTesting loaded model:")
print(f"Text: {test_sentence}")
print("Found entities:")
for ent in doc.ents:
    print(f"  {ent.text}: {ent.label_}")

new_text = "The company's earnings report exceeded expectations."
predicted_sentiment = predict_sentiment(new_text)

print(f"\nNew text: {new_text}")
print(f"Predicted sentiment: {predicted_sentiment}")

# Test text
test_text_loaded = "Apple CEO Tim Cook introduced the new iPhone model at a conference held in San Francisco."

# Test with selected hardware
hardware_loaded = 'cpu'  # can be changed to 'gpu'
print(f"\nTesting loaded model with {hardware_loaded.upper()}:")

# Make predictions with the loaded model
results_loaded = predict_entities_loaded(test_text_loaded, bert_model, bert_tokenizer, hardware_loaded)

# Show results
print("\nTest text:", test_text_loaded)
print("\nFound entities:")
for token, label in results_loaded:
    print(f"{token}: {label}")