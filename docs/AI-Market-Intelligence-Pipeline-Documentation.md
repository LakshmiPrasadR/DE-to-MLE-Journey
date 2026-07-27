# AI Market Intelligence Pipeline — Build Documentation
**Date built:** July 27, 2026
**Author:** Lakshmi Prasad R
**Project goal:** Automatically pull news on AI-related topics, store it in a searchable knowledge base, and let a user ask natural-language questions and get grounded, source-cited answers — a working RAG (Retrieval-Augmented Generation) system.

---

## 1. Big Picture: What We Built

A 4-stage pipeline:

```
[NewsAPI] → INGEST → TRANSFORM/CLEAN → EMBED & STORE → RETRIEVE & GENERATE (RAG)
                                                                    ↓
                                                          [User asks a question]
                                                                    ↓
                                                        [Grounded answer + sources]
```

**In plain English for explaining to someone non-technical:**
"I built a system that automatically collects news articles, understands their meaning using AI, stores that understanding in a searchable database, and then lets anyone ask a question in plain English — the system finds the most relevant articles and uses an AI model to write an answer based only on those real articles, with sources cited. It won't make things up — if the data doesn't have the answer, it says so."

**Why this matters (the "problem it solves"):** Instead of manually reading dozens of news articles to stay updated on a topic, one query gives you a synthesized, sourced answer. This is the same core pattern (RAG) used in real products like customer support bots, internal company knowledge search, and research assistants.

---

## 2. Repository Structure

```
DE-to-MLE-Journey/
├── ingestion/
│   └── fetch_news.py       # Stage 1: Pull raw data from NewsAPI
├── transform/
│   └── clean_articles.py   # Stage 2: Clean and standardize the data
├── embed/
│   └── embed_and_store.py  # Stage 3: Convert text to vectors, store in DB
├── api/
│   └── query.py            # Stage 4: Ask questions, get AI answers
├── data/
│   ├── news_*.json         # Raw pulled articles
│   ├── cleaned/             # Cleaned articles
│   └── vector_store/        # ChromaDB vector database files
├── requirements.txt         # Python packages needed
├── .env                     # API keys (NEVER committed to GitHub)
└── .gitignore                # Tells git to ignore .env
```

This is a standard **modular pipeline structure** — each stage is a separate, independent script with a single responsibility. This is a real industry pattern (separation of concerns) that makes pipelines easier to debug, test, and eventually orchestrate.

---

## 3. Stage-by-Stage Breakdown

### Stage 1 — Ingestion: `ingestion/fetch_news.py`

**What it does:** Calls the NewsAPI to pull recent articles across five AI-related topics and saves the raw results as a timestamped JSON file.

```python
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
```

**Line-by-line explanation:**
| Code | Purpose |
|---|---|
| `load_dotenv()` | Loads secret values (like API keys) from the `.env` file into the environment, so we never hardcode secrets in code |
| `os.getenv("NEWSAPI_KEY")` | Safely retrieves the API key without exposing it in the source code |
| `fetch_news(query, page_size)` | A reusable function — takes a search topic and how many articles to fetch, calls NewsAPI, returns a list of articles |
| `response.raise_for_status()` | If the API call fails (e.g., bad key, rate limit), this immediately throws a clear error instead of silently continuing with broken data |
| `for topic in topics:` loop | Runs the fetch function once per topic, building a broader, more diverse dataset than a single generic search |
| `all_articles.extend(...)` | Combines results from every topic into one master list |
| `json.dump(...)` | Saves everything to a timestamped file so every run is preserved and traceable — useful for debugging and for showing pipeline history |

**Concept to explain to others:** *API integration with error handling and reusable functions* — a foundational data engineering skill.

---

### Stage 2 — Transform: `transform/clean_articles.py`

**What it does:** Loads the most recent raw file, removes incomplete records, and reshapes each article into a clean, consistent format.

```python
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
```

**Line-by-line explanation:**
| Code | Purpose |
|---|---|
| `glob.glob("data/news_*.json")` | Finds every raw file that matches the naming pattern — lets the script automatically find files without hardcoding filenames |
| `max(files, key=os.path.getctime)` | Picks the most recently created file — so the pipeline always processes the latest data pull |
| `if not a.get("title") or not a.get("description"): continue` | **Data validation** — skips any article missing essential fields, preventing broken data downstream |
| `a["source"]["name"]` | NewsAPI nests the source name inside a sub-object; we flatten it out into a simple field — this is **schema standardization** |
| `.strip()` | Removes stray whitespace from text fields — small but standard data-cleaning practice |

**Concept to explain to others:** *Data validation and schema standardization* — this is the "T" (Transform) in ETL, a core data engineering concept.

---

### Stage 3 — Embed & Store: `embed/embed_and_store.py`

**What it does:** Converts each article's text into a numerical vector (embedding) that captures its *meaning*, then stores those vectors in a local vector database for fast similarity search.

```python
import json
import chromadb
from sentence_transformers import SentenceTransformer

def load_cleaned_articles():
    with open("data/cleaned/articles_cleaned.json") as f:
        return json.load(f)

def main():
    articles = load_cleaned_articles()

    print("Loading embedding model...")
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
    print(f"Stored {len(texts)} articles in vector DB")

if __name__ == "__main__":
    main()
```

**Line-by-line explanation:**
| Code | Purpose |
|---|---|
| `SentenceTransformer("all-MiniLM-L6-v2")` | Loads a free, pre-trained AI model that converts text into a list of numbers (a "vector") representing its meaning. Runs locally — no API cost |
| `chromadb.PersistentClient(path=...)` | Creates a local vector database that saves to disk (persists between runs), instead of an in-memory-only store |
| `collection.get_or_create_collection(...)` | A "collection" is like a table — this creates it if it doesn't exist yet, or reuses it if it does |
| `combined_text = f"{title}. {description}"` | Combines title + description into one text block per article — this is what actually gets embedded |
| `model.encode(texts)` | The core AI step — turns all article texts into embedding vectors in one batch (efficient) |
| `collection.upsert(...)` | Saves (inserts or updates) the vectors, original text, and metadata together, so later we can retrieve both the match *and* its source info |

**Concept to explain to others:** *Vector embeddings and semantic search* — this is what makes the system "understand" meaning rather than just matching keywords. Explain it like: "Instead of searching for exact words, the AI compares the *meaning* of your question to the *meaning* of each article."

---

### Stage 4 — Retrieve & Generate (RAG): `api/query.py`

**What it does:** Takes a user's question, finds the most relevant articles (retrieval), then sends those articles + the question to an LLM to generate a grounded answer (generation).

```python
import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client_ai = Groq(api_key=os.getenv("GROQ_API_KEY"))
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="data/vector_store")
collection = chroma_client.get_collection(name="news_articles")

def retrieve(query, n_results=3):
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results

def generate_answer(query, retrieved):
    context_chunks = []
    for doc, meta in zip(retrieved["documents"][0], retrieved["metadatas"][0]):
        context_chunks.append(f"- {doc} (Source: {meta['source']}, {meta['published_at']})")
    context = "\n".join(context_chunks)

    prompt = f"""Answer the question using only the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {query}

Answer:"""

    response = client_ai.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    query = input("Ask a question about recent AI news: ")
    retrieved = retrieve(query)
    answer = generate_answer(query, retrieved)
    print("\n--- ANSWER ---")
    print(answer)
    print("\n--- SOURCES ---")
    for meta in retrieved["metadatas"][0]:
        print(f"- {meta['title']} ({meta['source']}, {meta['url']})")
```

**Line-by-line explanation:**
| Code | Purpose |
|---|---|
| `retrieve(query, n_results=3)` | **The "R" in RAG.** Embeds the user's question the same way we embedded articles, then asks the vector DB: "which 3 stored articles are closest in meaning to this question?" |
| `collection.query(query_embeddings=..., n_results=3)` | ChromaDB's similarity search — compares vectors mathematically to find the closest matches |
| `generate_answer(query, retrieved)` | **The "G" in RAG.** Takes the retrieved articles and builds a prompt that instructs the LLM to answer *only* using that context |
| The `prompt` f-string | This is **prompt engineering** — explicitly telling the model "use only the context below" and "if it's not there, say so" is what prevents hallucination |
| `client_ai.chat.completions.create(...)` | The actual API call to Groq's hosted LLM (Llama 3.3 70B model), sending the prompt and getting a generated response back |
| Final `for meta in ...` loop | Prints out the source articles used, so the answer is **traceable and verifiable** — critical for trust in real AI systems |

**Concept to explain to others:** *This is RAG — Retrieval-Augmented Generation.* Explain it as: "Step 1: find the most relevant real information. Step 2: hand that information to an AI model and ask it to answer using only that information. This is why the system said 'I don't know' when the data didn't have the answer, instead of making something up — that's the entire point of RAG over a plain chatbot."

---

## 4. Key Engineering Decisions (good interview talking points)

| Decision | Why it matters |
|---|---|
| Separate scripts per stage (ingest/transform/embed/query) | Mirrors real-world pipeline architecture — each stage can be run, tested, debugged, or scheduled independently |
| `.env` + `.gitignore` for API keys | Security best practice — secrets never get committed to version control |
| Local embedding model (not an API) | No cost, no external dependency for this step, faster iteration |
| Grounded prompt instructing "use only context" | Deliberate hallucination prevention — a real production concern with LLMs |
| Timestamped raw data files | Preserves pipeline history/traceability instead of overwriting data each run |
| Switch from Claude API to Groq | Practical adaptability — when one API had a billing blocker, pivoted to a free alternative with minimal code change, since both use a similar chat-completion pattern |

---

## 5. What's Next (not built yet)

- **Orchestration (Airflow):** automate this to run daily instead of manually
- **API wrapper (FastAPI):** turn `query.py` into a real web service/endpoint
- **Deployment (Docker + AWS):** containerize and host it
- **Documentation polish:** architecture diagram, README for the repo itself

---

## 6. One-Sentence Summary (for a resume bullet or quick verbal explanation)

*"Built an end-to-end RAG pipeline that ingests live news via API, cleans and validates the data, generates vector embeddings for semantic search, and serves grounded, source-cited answers to natural-language questions using an LLM — demonstrating data engineering fundamentals combined with applied AI system design."*
