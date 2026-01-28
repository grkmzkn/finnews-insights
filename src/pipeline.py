"""
Financial News Analysis Pipeline
Fetches live news and analyzes with trained models.
"""
import os
import json
import pandas as pd
from datetime import datetime

# Import news fetcher
from news_fetcher import NewsFetcher

# Import helper functions
import helpful_functions as hf


def analyze_news_with_models(articles, models, use_models=['sentiment', 'spacy']):
    """
    Analyze news articles with selected models.
    
    Args:
        articles: List of article dictionaries from news_fetcher
        models: Dictionary containing loaded models (from load_models())
        use_models: List of models to use ['sentiment', 'spacy', 'bert', 'gemini']
        
    Returns:
        List of analysis results
    """
    # Unpack models
    sentiment_model = models['sentiment_model']
    spacy_model = models['spacy_model']
    bert_model = models['bert_model']
    bert_tokenizer = models['bert_tokenizer']
    tfidf = models['tfidf']
    id2label = models['id2label']
    
    results = []
    total = len(articles)
    
    print(f"\n{'='*70}")
    print(f"📊 ANALYZING {total} ARTICLES")
    print(f"{'='*70}\n")
    
    for i, article in enumerate(articles, 1):
        print(f"[{i}/{total}] {article['title'][:60]}...")
        
        # Combine title and description for analysis
        text = f"{article['title']}. {article['description']}"
        
        analysis = {}
        
        # 1. Sentiment Analysis (Traditional ML)
        if 'sentiment' in use_models:
            try:
                sentiment = hf.predict_sentiment(tfidf, sentiment_model, text)
                analysis['sentiment'] = sentiment
                print(f"  📊 Sentiment: {sentiment.upper()}")
            except Exception as e:
                analysis['sentiment'] = f"Error: {str(e)}"
                print(f"  ❌ Sentiment error: {str(e)}")
        
        # 2. SpaCy NER
        if 'spacy' in use_models:
            try:
                spacy_entities = hf.predict_entities_spacy(spacy_model, text)
                entities = [{'text': ent_text, 'label': ent_label} for ent_text, ent_label in spacy_entities]
                analysis['spacy_entities'] = entities
                if entities:
                    print(f"  🏷️  SpaCy Entities ({len(entities)}):")
                    for ent in entities:
                        print(f"      - {ent['text']}: {ent['label']}")
                else:
                    print(f"  🏷️  SpaCy Entities: None found")
            except Exception as e:
                analysis['spacy_entities'] = []
                print(f"  ❌ SpaCy error: {str(e)}")
        
        # 3. BERT NER
        if 'bert' in use_models:
            try:
                bert_entities = hf.predict_entities_bert(id2label, text, bert_model, bert_tokenizer, 'cpu')
                analysis['bert_entities'] = [
                    {'text': token, 'label': label} for token, label in bert_entities
                ]
                if bert_entities:
                    print(f"  🏷️  BERT Entities ({len(bert_entities)}):")
                    for token, label in bert_entities:
                        print(f"      - {token}: {label}")
                else:
                    print(f"  🏷️  BERT Entities: None found")
            except Exception as e:
                analysis['bert_entities'] = []
                print(f"  ❌ BERT error: {str(e)}")
        
        # 4. Gemini Analysis
        if 'gemini' in use_models:
            try:
                gemini_result = hf.analyze_with_gemini(text)
                analysis['gemini_sentiment'] = gemini_result.get('sentiment', 'N/A')
                analysis['gemini_entities'] = gemini_result.get('entities', [])
                print(f"  🤖 Gemini Sentiment: {gemini_result.get('sentiment', 'N/A')}")
                gemini_entities = gemini_result.get('entities', [])
                if gemini_entities:
                    print(f"  🤖 Gemini Entities ({len(gemini_entities)}):")
                    for ent in gemini_entities:
                        print(f"      - {ent}")
                else:
                    print(f"  🤖 Gemini Entities: None found")
            except Exception as e:
                analysis['gemini_sentiment'] = f"Error: {str(e)}"
                analysis['gemini_entities'] = []
                print(f"  ❌ Gemini error: {str(e)}")
        
        # Store result
        result = {
            'article': {
                'title': article['title'],
                'source': article['source'],
                'url': article['url'],
                'published': article['published'],
                'description': article['description']
            },
            'analysis': analysis,
            'analyzed_at': datetime.now().isoformat()
        }
        results.append(result)
        print()
    
    return results

def run_pipeline(max_articles=10, use_models=['sentiment', 'spacy', 'bert', 'gemini']):
    """
    Run the complete analysis pipeline.
    
    Args:
        max_articles: Maximum number of articles to fetch per source
        use_models: List of models to use ['sentiment', 'spacy', 'bert', 'gemini']
        
    Returns:
        List of analysis results
    """
    
    # Step 0: Load models
    print(f"🔧 STEP 0: LOADING MODELS")
    print(f"\n{'='*70}")
    sentiment_model, spacy_model, bert_model, bert_tokenizer, tfidf, id2label = hf.load_models()
    
    # Pack models into dictionary
    models = {
        'sentiment_model': sentiment_model,
        'spacy_model': spacy_model,
        'bert_model': bert_model,
        'bert_tokenizer': bert_tokenizer,
        'tfidf': tfidf,
        'id2label': id2label
    }
    
    print("✅ Models loaded successfully")
    print(f"\n{'='*70}")
    
    # Step 1: Fetch news
    print(f"📰 STEP 1: FETCHING NEWS FROM RSS FEEDS")
    print(f"{'='*70}\n")
    
    fetcher = NewsFetcher()
    articles = fetcher.fetch_from_rss('all', max_articles=max_articles)
    
    if not articles:
        print("❌ No articles fetched. Check your internet connection.")
        return []
    
    print(f"\n✅ Fetched {len(articles)} articles from RSS feeds")
    
    # Step 2: Analyze articles
    print(f"🔬 STEP 2: ANALYZING WITH MODELS")
    print(f"{'='*70}")
    
    results = analyze_news_with_models(articles, models, use_models=use_models)
    
    print(f"\n{'='*70}")
    print(f"✅ PIPELINE COMPLETE!")
    print(f"{'='*70}\n")
    
    return results

if __name__ == "__main__":
    results = run_pipeline(
        max_articles=5,
        use_models=['sentiment', 'spacy', 'bert', 'gemini']  # Use all models
    )
    
    print(f"Total analyzed: {len(results)} articles")
