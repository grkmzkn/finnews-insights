"""
News Fetcher Module
Fetches financial news from free RSS feeds.
"""
import feedparser
from datetime import datetime
from typing import List, Dict
import time

class NewsFetcher:
    """Fetch news from free RSS feeds."""
    
    def __init__(self):
        """Initialize news fetcher with RSS feeds."""
        # RSS Feeds (completely free, no API key needed)
        self.rss_feeds = {
            'reuters_business': 'https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best',
            'cnbc_top': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'cnbc_us': 'https://www.cnbc.com/id/15837362/device/rss/rss.html',
            'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss',
            'marketwatch': 'http://feeds.marketwatch.com/marketwatch/topstories/',
            'investing_stock': 'https://www.investing.com/rss/news.rss',
            'yahoo_finance': 'https://finance.yahoo.com/news/rssindex',
        }
    
    def fetch_from_rss(self, source='all', max_articles=10) -> List[Dict]:
        """
        Fetch news from RSS feeds (completely free).
        
        Args:
            source: RSS source name or 'all' for all sources
            max_articles: Maximum number of articles per source
            
        Returns:
            List of articles with title, description, link, published date
        """
        articles = []
        
        feeds_to_fetch = self.rss_feeds if source == 'all' else {source: self.rss_feeds.get(source)}
        
        for feed_name, feed_url in feeds_to_fetch.items():
            if feed_url is None:
                continue
                
            try:
                print(f"Fetching from {feed_name}...")
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:max_articles]:
                    article = {
                        'source': feed_name,
                        'title': entry.get('title', ''),
                        'description': entry.get('summary', entry.get('description', '')),
                        'content': entry.get('content', [{}])[0].get('value', '') if entry.get('content') else '',
                        'url': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'fetched_at': datetime.now().isoformat()
                    }
                    articles.append(article)
                    
                print(f"  ✓ Fetched {len(feed.entries[:max_articles])} articles")
                time.sleep(0.5)  # Be nice to servers
                
            except Exception as e:
                print(f"  ✗ Error fetching from {feed_name}: {str(e)}")
        
        return articles


if __name__ == "__main__":
    
    fetcher = NewsFetcher()
    
    # Fetch only from RSS feeds (no API key needed)
    print("Testing RSS feeds...")
    articles = fetcher.fetch_from_rss('all', max_articles=3)
    
    print(f"\nFetched {len(articles)} articles")
    if articles:
        print("\nFirst article:")
        print(f"  Title: {articles[0]['title']}")
        print(f"  Source: {articles[0]['source']}")
        print(f"  URL: {articles[0]['url']}")