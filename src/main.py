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
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

###########################################################################
# Configure Gemini
###########################################################################
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

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
        
        # Markdown code block'u temizle
        if "```json" in text_response:
            # İlk ```json ve son ``` arasındaki içeriği al
            json_content = text_response.split("```json")[1].split("```")[0]
        elif "```" in text_response:
            # İlk ``` ve son ``` arasındaki içeriği al
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

###########################################################################
# Load models
###########################################################################
# Load the TF-IDF vectorizer
with open("./models/sentiment_model/tfidf_vector.pkl", "rb") as file:
    tfidf = pickle.load(file)

# Load labels
with open('./data/label.json', 'r') as f:
    label_dict = json.load(f)

label2id = label_dict  # mappings from label.json
id2label = {v: k for k, v in label2id.items()}

# Load Sentiment model
sentiment_model = joblib.load("./models/sentiment_model/sentiment_model.pkl")

# Load SpaCy model
spacy_model = spacy.load("./models/spacy_model")

# Load Bert model
bert_tokenizer = AutoTokenizer.from_pretrained("./models/bert_model")
bert_model = AutoModelForTokenClassification.from_pretrained("./models/bert_model")
############################################################################

###########################################################################
# Preprocess
###########################################################################
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
test_sentence = "Apple Inc. reported a 12% increase in revenue during the second quarter of 2024."
print(test_sentence)
print("**" * 50)

# Traditional ML Sentiment prediction
predicted_sentiment = predict_sentiment(test_sentence)
print(f"\n1. Traditional ML Sentiment Model")
print(f"Predicted sentiment: {predicted_sentiment}")

# SpaCy prediction
print(f"\n2. SpaCy NER Model")
doc = spacy_model(test_sentence)
print("Found entities:")
for ent in doc.ents:
    print(f"  {ent.text}: {ent.label_}")

# BERT prediction
print(f"\n3. BERT Model")
hardware_loaded = 'cpu'
results_loaded = predict_entities_loaded(test_sentence, bert_model, bert_tokenizer, hardware_loaded)
print("Found entities:")
for token, label in results_loaded:
    print(f"  {token}: {label}")

# Gemini prediction
print(f"\n4. Gemini Model")
gemini_results = analyze_with_gemini(test_sentence)
print("Sentiment:", gemini_results['sentiment'])
print("Found entities:")
for entity in gemini_results['entities']:
    print(f"  {entity['text']}: {entity['type']}")