import re
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pickle
import json
import joblib
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification

import torch
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

def preprocess_text(text: str) -> str:
    """
    Preprocess the input text by applying various cleaning and normalization steps.
    
    Args:
        text (str): Raw input text to preprocess
        
    Returns:
        str: Cleaned and preprocessed text
    """
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

def load_models():
    """
    Load all trained models and necessary components for analysis.
    
    Returns:
        tuple: A tuple containing:
            - sentiment_model: Trained sentiment classification model
            - spacy_model: Loaded SpaCy NER model
            - bert_model: Loaded BERT model for token classification
            - bert_tokenizer: BERT tokenizer
            - tfidf: TF-IDF vectorizer for sentiment analysis
            - id2label: Dictionary mapping label IDs to label names
    """
    with open("./models/sentiment_model/tfidf_vector.pkl", "rb") as file:
        tfidf = pickle.load(file)
    
    with open('./data/label.json', 'r') as f:
        label_dict = json.load(f)

    label2id = label_dict
    id2label = {v: k for k, v in label2id.items()}

    sentiment_model = joblib.load("./models/sentiment_model/sentiment_model.pkl")
    spacy_model = spacy.load("./models/spacy_model")

    bert_tokenizer = AutoTokenizer.from_pretrained("./models/bert_model")
    bert_model = AutoModelForTokenClassification.from_pretrained("./models/bert_model")

    return sentiment_model, spacy_model, bert_model, bert_tokenizer, tfidf, id2label
    
def predict_sentiment(tfidf, sentiment_model, text):
    """
    Predict sentiment of the given text using traditional ML model.
    
    Args:
        tfidf: Pre-trained TF-IDF vectorizer
        sentiment_model: Trained sentiment classification model
        text (str): Input text to analyze
        
    Returns:
        str: Predicted sentiment label ('positive', 'negative', or 'neutral')
    """
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

def predict_entities_spacy(spacy_model, text):
    """
    Identifies entities in the text using SpaCy model.
    
    Args:
        spacy_model: Loaded SpaCy model
        text (str): Text to process
        
    Returns:
        List of tuples (entity_text, entity_label)
    """
    doc = spacy_model(text)
    results = []
    for ent in doc.ents:
        results.append((ent.text, ent.label_))
    return results

def predict_entities_bert(id2label, text, model, tokenizer, hardware='cpu'):
    """
    Identifies entities in the text using a loaded BERT model.
    
    Args:
        id2label (dict): Dictionary mapping label IDs to label names
        text (str): Text to process
        model: Loaded BERT model for token classification
        tokenizer: BERT tokenizer
        hardware (str): Hardware to use - 'cpu' or 'gpu'. Default is 'cpu'
        
    Returns:
        list: List of tuples (token, label) for identified entities
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

def analyze_with_gemini(text: str) -> dict:
    """
    Analyze text using Gemini model for sentiment and entity extraction.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        Dict containing sentiment and entities
    """
    prompt = f"""
    Analyze the following financial news text and provide:
    1. Sentiment (POSITIVE, NEGATIVE, or NEUTRAL)
    2. Entities (DATE, PERSON, ORG, PERCENT, MONEY)
    
    Text: {text}
    
    Format the response as a JSON with the following structure:
    {{
        "sentiment": "POSITIVE/NEGATIVE/NEUTRAL",
        "entities": [
            {{
                "text": "entity name",
                "type": "entity type"
            }}
        ]
    }}
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        text_response = response.text
        
        if "```json" in text_response:
            json_content = text_response.split("```json")[1].split("```")[0]
        elif "```" in text_response:
            json_content = text_response.split("```")[1].split("```")[0]
        else:
            json_content = text_response
            
        # Temizlenmiş JSON string'ini parse et
        json_content = json_content.strip()
        result = json.loads(json_content)
        return result
    except Exception as e:
        print(f"Error in Gemini analysis: {str(e)}")
        print(f"Raw response: {response.text}")  # Debug için ham yanıtı yazdır
        return {"sentiment": "NEUTRAL", "entities": []}