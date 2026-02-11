# data_sources/news.py
from newsapi import NewsApiClient
import os

newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))

def get_market_news(query="bitcoin"):
    articles = newsapi.get_everything(q=query, language="en", page_size=5)
    return articles['articles']
