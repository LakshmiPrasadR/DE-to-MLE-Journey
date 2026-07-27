import requests
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NEWSAPI_KEY")

def fetch_news(query="artificial intelligence", page_size=20):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": API_KEY
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()["articles"]

if __name__ == "__main__":
    topics = [
        "artificial intelligence regulation",
        "AI policy government",
        "AI safety",
        "generative AI",
        "AI industry news"
    ]
    all_articles = []
    for topic in topics:
        print(f"Fetching: {topic}")
        all_articles.extend(fetch_news(query=topic, page_size=20))

    os.makedirs("data", exist_ok=True)
    filename = f"data/news_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(filename, "w") as f:
        json.dump(all_articles, f, indent=2)
    print(f"Saved {len(all_articles)} articles to {filename}")