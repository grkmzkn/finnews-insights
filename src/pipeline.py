"""
Financial News Analysis Pipeline
Fetches live news and analyzes with trained models.
"""
import os
import json
import pandas as pd
from datetime import datetime
import hashlib
import time
import requests

# Import news fetcher
from news_fetcher import NewsFetcher

# Import helper functions
import helpful_functions as hf

# Cache file path
CACHE_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'news_cache.json')


def get_article_hash(article):
    """Create unique hash for an article based on title and URL."""
    unique_str = f"{article['title']}|{article['url']}"
    return hashlib.md5(unique_str.encode('utf-8')).hexdigest()


def load_cache():
    """Load cache of analyzed articles."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load cache: {e}")
    return {'analyzed_hashes': [], 'total_analyzed': 0}


def save_cache(cache_data):
    """Save cache of analyzed articles."""
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Could not save cache: {e}")


def send_to_telegram(results, bot_token=None, chat_id=None):
    """Send analysis results to Telegram channel/chat. Each article sent as separate message."""
    if not bot_token or not chat_id:
        print("⚠️ Telegram not configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)")
        return
    
    if not results:
        return
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    success_count = 0
    
    # Send each article as separate message
    for i, result in enumerate(results, 1):
        try:
            article = result['article']
            analysis = result['analysis']
            
            # Sentiment emoji
            sentiment = analysis.get('sentiment', 'neutral').lower()
            emoji = "📈" if sentiment == 'positive' else "📉" if sentiment == 'negative' else "➖"
            
            # Build message for this article
            message = f"*{i}/{len(results)} - {emoji} {article['title']}*\n\n"
            message += f"📰 *Kaynak:* {article['source']}\n"
            message += f"💭 *Sentiment:* {sentiment.upper()}\n"
            
            # SpaCy Entities
            spacy_entities = analysis.get('spacy_entities', [])
            if spacy_entities:
                message += f"\n🏷️ *SpaCy Entities ({len(spacy_entities)}):*\n"
                for ent in spacy_entities:
                    message += f"  • {ent['text']}: `{ent['label']}`\n"
            
            # BERT Entities
            bert_entities = analysis.get('bert_entities', [])
            if bert_entities:
                message += f"\n🤖 *BERT Entities ({len(bert_entities)}):*\n"
                for ent in bert_entities:
                    message += f"  • {ent['text']}: `{ent['label']}`\n"
            
            # Gemini Analysis
            gemini_sentiment = analysis.get('gemini_sentiment')
            if gemini_sentiment and gemini_sentiment != 'N/A':
                message += f"\n🧠 *Gemini Sentiment:* {gemini_sentiment}\n"
            
            gemini_entities = analysis.get('gemini_entities', [])
            if gemini_entities:
                message += f"🧠 *Gemini Entities ({len(gemini_entities)}):*\n"
                for ent in gemini_entities:
                    if isinstance(ent, dict):
                        text = ent.get('text', ent.get('entity', 'N/A'))
                        ent_type = ent.get('type', ent.get('label', 'N/A'))
                        message += f"  • {text}: `{ent_type}`\n"
                    else:
                        message += f"  • {str(ent)}\n"
            
            message += f"\n🔗 [Haberi Oku]({article['url']})"
            
            # Check message length (Telegram limit: 4096 characters)
            if len(message) > 4096:
                message = message[:4090] + "...\n\n_[Mesaj kesildi]_"
            
            # Send message
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                success_count += 1
                print(f"✅ [{i}/{len(results)}] Telegram'a gönderildi: {article['title'][:50]}...")
            else:
                print(f"⚠️ [{i}/{len(results)}] Telegram gönderimi başarısız: {response.text}")
            
            # Small delay between messages to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ [{i}/{len(results)}] Haber gönderimi hatası: {e}")
    
    print(f"\n✅ Toplam {success_count}/{len(results)} mesaj başarıyla gönderildi")


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

def run_pipeline(models, max_articles=10, use_models=['sentiment', 'spacy']):
    """
    Run the complete analysis pipeline.
    Fetches news and analyzes only new (not previously analyzed) articles.
    
    Args:
        models: Dictionary containing loaded models
        max_articles: Maximum number of articles to fetch per source
        use_models: List of models to use ['sentiment', 'spacy', 'bert', 'gemini']
        
    Returns:
        List of analysis results for new articles
    """
    
    print(f"\n{'='*70}")
    print(f"🔍 CHECKING FOR NEWS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # Step 1: Fetch news
    print(f"📰 STEP 1: FETCHING NEWS FROM RSS FEEDS")
    print(f"{'='*70}\n")
    
    fetcher = NewsFetcher()
    articles = fetcher.fetch_from_rss('all', max_articles=max_articles)
    
    if not articles:
        print("❌ No articles fetched. Check your internet connection.")
        return []
    
    print(f"\n✅ Fetched {len(articles)} articles from RSS feeds")
    
    # Step 2: Load cache and filter new articles
    print(f"\n🔎 STEP 2: CHECKING FOR NEW ARTICLES")
    print(f"{'='*70}\n")
    
    cache = load_cache()
    analyzed_hashes = set(cache.get('analyzed_hashes', []))
    
    new_articles = []
    for article in articles:
        article_hash = get_article_hash(article)
        if article_hash not in analyzed_hashes:
            new_articles.append(article)
            analyzed_hashes.add(article_hash)
    
    if not new_articles:
        print(f"ℹ️  No new articles found (all {len(articles)} already analyzed)")
        print(f"\n{'='*70}")
        print(f"✅ CHECK COMPLETE - No new articles to analyze")
        print(f"{'='*70}\n")
        return []
    
    print(f"🆕 Found {len(new_articles)} NEW article(s) out of {len(articles)} total")
    
    # Step 3: Analyze only new articles
    print(f"\n🔬 STEP 3: ANALYZING NEW ARTICLES")
    print(f"{'='*70}")
    
    results = analyze_news_with_models(new_articles, models, use_models=use_models)
    
    # Step 4: Update cache
    cache_data = {
        'analyzed_hashes': list(analyzed_hashes),
        'last_check': datetime.now().isoformat(),
        'total_analyzed': len(analyzed_hashes)
    }
    save_cache(cache_data)
    
    # Step 5: Save results
    if results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'results', f'analysis_{timestamp}.json')
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {os.path.basename(results_file)}")
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")
    
    print(f"\n{'='*70}")
    print(f"✅ PIPELINE COMPLETE - {len(new_articles)} new article(s) analyzed")
    print(f"{'='*70}\n")
    
    # Send to Telegram if configured
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if results and (bot_token and chat_id):
        send_to_telegram(results, bot_token, chat_id)
    
    return results

if __name__ == "__main__":
    # Configuration
    CHECK_INTERVAL_MINUTES = 5
    MAX_ARTICLES = 1
    USE_MODELS = ['sentiment', 'spacy', 'bert', 'gemini']
    
    # Load models once at the beginning
    print(f"🔧 LOADING MODELS...")
    print(f"{'='*70}\n")
    
    sentiment_model, spacy_model, bert_model, bert_tokenizer, tfidf, id2label = hf.load_models()
    
    models = {
        'sentiment_model': sentiment_model,
        'spacy_model': spacy_model,
        'bert_model': bert_model,
        'bert_tokenizer': bert_tokenizer,
        'tfidf': tfidf,
        'id2label': id2label
    }
    
    print("✅ Models loaded successfully\n")
    
    try:
        while True:
            run_pipeline(models=models, max_articles=MAX_ARTICLES, use_models=USE_MODELS)
            print(f"\n⏰ Next check in {CHECK_INTERVAL_MINUTES} minutes...")
            time.sleep(CHECK_INTERVAL_MINUTES * 60)
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user")
