import json
import chromadb
from sentence_transformers import SentenceTransformer

def load_cleaned_articles():
    with open("data/cleaned/articles_cleaned.json") as f:
        return json.load(f)

def main():
    articles = load_cleaned_articles()

    print("Loading embedding model (first run downloads it, ~90MB)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path="data/vector_store")
    collection = client.get_or_create_collection(name="news_articles")

    texts, ids, metadatas = [], [], []
    for i, article in enumerate(articles):
        combined_text = f"{article['title']}. {article['description']}"
        texts.append(combined_text)
        ids.append(f"article_{i}")
        metadatas.append({
            "title": article["title"],
            "source": article["source"],
            "url": article["url"],
            "published_at": article["published_at"]
        })

    print(f"Embedding {len(texts)} articles...")
    embeddings = model.encode(texts).tolist()

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    print(f"Stored {len(texts)} articles in vector DB at data/vector_store")

if __name__ == "__main__":
    main()