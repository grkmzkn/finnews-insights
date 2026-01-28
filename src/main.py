import helpful_functions as hf
import os

test_sentence = "The collapse of two US companies could be a sign of wider problems in the financial system and \"alarm bells\" were ringing, the governor of the Bank of England has said."
print(test_sentence)
print("**" * 50)

###########################################################################
# Load models
sentiment_model, spacy_model, bert_model, bert_tokenizer, tfidf, id2label = hf.load_models()

###########################################################################
# Preprocess
test_sentence = hf.preprocess_text(test_sentence)

###########################################################################
# Sentiment Analysis
predicted_sentiment = hf.predict_sentiment(tfidf, sentiment_model, test_sentence)
print(f"\n1. Traditional ML Sentiment Model")
print(f"Predicted sentiment: {predicted_sentiment}")

###########################################################################
# BERT NER Prediction
results_bert = hf.predict_entities_bert(id2label, test_sentence, bert_model, bert_tokenizer, hardware='cpu')
print(f"\n2. BERT NER Model")
print("Found entities:")
for token, label in results_bert:
    print(f"  {token}: {label}")

###########################################################################
# SpaCy NER Prediction
results_spacy = hf.predict_entities_spacy(spacy_model, test_sentence)
print(f"\n3. SpaCy NER Model")
print("Found entities:")
for entity_text, entity_label in results_spacy:
    print(f"  {entity_text}: {entity_label}")

###########################################################################
# Gemini prediction
print(f"\n4. Gemini Model")
gemini_results = hf.analyze_with_gemini(test_sentence)
print("Sentiment:", gemini_results['sentiment'])
print("Found entities:")
for entity in gemini_results['entities']:
    print(f"  {entity['text']}: {entity['type']}")