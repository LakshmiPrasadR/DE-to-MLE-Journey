import json
import glob
import os

def load_latest_raw_file():
    files = glob.glob("data/news_*.json")
    latest = max(files, key=os.path.getctime)
    with open(latest) as f:
        return json.load(f)

def clean_articles(articles):
    cleaned = []
    for a in articles:
        if not a.get("title") or not a.get("description"):
            continue  # skip incomplete articles
        cleaned.append({
            "title": a["title"].strip(),
            "description": a["description"].strip(),
            "content": (a.get("content") or "").strip(),
            "source": a["source"]["name"],
            "url": a["url"],
            "published_at": a["publishedAt"]
        })
    return cleaned

if __name__ == "__main__":
    raw = load_latest_raw_file()
    cleaned = clean_articles(raw)
    os.makedirs("data/cleaned", exist_ok=True)
    with open("data/cleaned/articles_cleaned.json", "w") as f:
        json.dump(cleaned, f, indent=2)
    print(f"Cleaned {len(cleaned)} out of {len(raw)} articles")