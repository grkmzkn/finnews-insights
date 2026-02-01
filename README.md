# Financial News Insights

Automated financial news analysis system with real-time monitoring, multi-model NLP analysis, and Telegram notifications.

<img width="716" height="601" alt="image" src="https://github.com/user-attachments/assets/f0e2157a-cb2d-4b83-8bbf-e78f923817dc" />


## Overview

Continuous monitoring service that fetches financial news from RSS feeds, analyzes them using multiple ML models, and sends notifications to Telegram. Features smart caching to avoid duplicate analysis.

**Key Features:**
- 🔄 **Real-time News Monitoring**: Fetches from multiple financial news sources
- 🤖 **Multi-Model Analysis**: BERT NER, SpaCy NER, Traditional ML Sentiment, Gemini AI (optional)
- 📱 **Telegram Notifications**: Instant alerts for new financial news
- 💾 **Smart Caching**: Analyzes only new articles, prevents duplicates
- ☁️ **Cloud-Ready**: Easy deployment to Render, Railway, or Hugging Face Spaces

## Quick Start

### Prerequisites
- Python 3.10+
- Telegram Bot Token ([get from @BotFather](https://t.me/BotFather))
- Telegram Chat ID

### Installation

```bash
# Clone repository
git clone https://github.com/grkmzkn/finnews-insights.git
cd finnews-insights

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# Set environment variables
cp .env.example .env
# Edit .env and add your Telegram credentials
```

### Configuration

Create `.env` file:
```bash
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
GEMINI_API_KEY=your_gemini_key_here  # Optional
```

### Run

```bash
# Start continuous monitoring
cd src
python pipeline.py
```

The service will check for new financial news every 30 minutes and send analysis to Telegram.

## Analysis Pipeline

The system analyzes each financial news article using:

1. **Sentiment Analysis** (Traditional ML)
   - TF-IDF + ML classifier
   - Categories: Positive, Negative, Neutral
   - Financial-specific preprocessing

2. **SpaCy NER**
   - Lightweight custom-trained model
   - Detects: Organizations, People, Dates, Money, Percentages
   - Fast inference

3. **BERT NER**
   - Transformer-based model
   - High accuracy for complex contexts
   - Better entity boundary detection

4. **Gemini AI** (Optional)
   - Advanced LLM analysis
   - Enhanced entity recognition
   - Contextual sentiment understanding

ResProject Structure

```
finnews-insights/
├── src/
│   ├── pipeline.py              # Main monitoring service
│   ├── news_fetcher.py         # RSS feed crawler
│   ├── helpful_functions.py    # Model loading & analysis
│   └── main.py                 # Single analysis example
├── models/
│   ├── bert_model/             # BERT NER model
│   ├── spacy_model/            # SpaCy NER model
│   └── sentiment_model/        # Traditional ML model
├── data/
│   ├── label.json              # Entity label mapping
│   ├── news_cache.json         # Analyzed articles cache
│   └── results/                # Analysis results (JSON)
└──How It Works

1. **News Fetching**: Monitors RSS feeds from Reuters, Bloomberg, CNBC, Yahoo Finance
2. **Deduplication**: Checks cache to skip already-analyzed articles
3. **Multi-Model Analysis**: Runs all enabled models on new articles
4. **Result Storage**: Saves JSON results to `data/results/`
5. **Telegram Notification**: Sends formatted analysis to your Telegram chat
6. **Repeat**: Waits 30 minutes and starts again

## Configuration Options

Edit `src/pipeline.py` (at the bottom):

```python
CHECK_INTERVAL_MINUTES = 30  # How often to check for news
MAX_ARTICLES = 10            # Articles per source per check
USE_MODELS = ['sentiment', 'spacy', 'bert', 'gemini']  # Models to use
```

**Model Options:**
- `sentiment`: Traditional ML sentiment analysis (fast)
- `spacy`: SpaCy NER (fast, lightweight)
- `bert`: BERT NER (accurate, slower)
- `gemini`: Gemini AI (requires API key, most accurate)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- GitHub: [@grkmzkn](https://github.com/grkmzkn)
