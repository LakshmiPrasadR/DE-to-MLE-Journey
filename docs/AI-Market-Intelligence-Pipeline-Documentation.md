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

## 5. Day 2 Addition: Orchestration with Apache Airflow
**Date built:** July 28, 2026

### 5.1 The Business Problem This Solves

Up to yesterday, the pipeline worked — but only if a human sat down and manually typed three commands in order:
```bash
python ingestion/fetch_news.py
python transform/clean_articles.py
python embed/embed_and_store.py
```

**In plain, layman's terms:** imagine you built a great vending machine, but every single day someone has to personally restock it by hand, in the right order, or it breaks. That's not a real business-ready system — it's a demo that depends on you remembering to run it.

**What we solved today:** we hired an automatic, tireless "manager" (Apache Airflow) whose only job is to run those three steps, in the correct order, every single day, on its own — and to raise a red flag immediately if any step fails, instead of silently producing bad or stale data.

**Why this matters for a real business:** stale data is a real cost. If a company's AI assistant is answering questions based on news from three weeks ago because nobody remembered to refresh it, that's a trust and reliability problem. Automated orchestration is what separates a "cool personal project" from "a system a business could actually depend on."

### 5.2 What Airflow Actually Is (Layman's Explanation)

Airflow is a **scheduling and monitoring tool**. Think of it like a project manager that:
1. Knows the exact list of tasks that need to happen (`fetch → clean → embed`)
2. Knows the correct order and won't let a later task start until the one before it finishes successfully
3. Runs everything automatically on a schedule (we set it to once a day)
4. Gives you a visual dashboard to see: did today's run succeed? Did anything fail? How long did each step take?

This "list of tasks + their order" is called a **DAG** (Directed Acyclic Graph) — a fancy term that just means "a flowchart of steps where you never loop backward."

### 5.3 Step-by-Step: What We Did Today

| Step | What we did | Why |
|---|---|---|
| 1 | Installed Apache Airflow (`pip install apache-airflow`) | Adds the orchestration tool to our project environment |
| 2 | Ran `airflow db migrate` | Airflow needs its own small internal database to keep track of run history, schedules, and logs — this sets that up |
| 3 | Ran `airflow standalone` | Starts Airflow's web dashboard (port 8080) and its background "scheduler" (the part that actually watches the clock and triggers runs) |
| 4 | Wrote a DAG file (`dags/ai_news_pipeline_dag.py`) | This is the actual "instruction sheet" telling Airflow what to run and in what order (details below) |
| 5 | Pointed Airflow at our project's `dags/` folder | By default Airflow looks in its own private folder — we redirected it to look inside our GitHub repo instead, so our DAG lives with the rest of our code, not hidden away |
| 6 | Logged into the Airflow web dashboard | Confirmed the system was healthy (database, scheduler, etc. all green) |
| 7 | Found our DAG in the dashboard and triggered it manually | A real-world test run, to prove it works before trusting the automatic daily schedule |
| 8 | Watched all 3 tasks turn green in the dashboard | Confirmed success — no manual commands needed, Airflow ran our entire pipeline by itself |
| 9 | Re-ran `python api/query.py` after the automated run | Proved the *automated* run produced real, usable, fresh data — not just "tasks turned green" but "the actual business output is correct" |
| 10 | Committed the DAG file to GitHub | Saved this new orchestration layer as part of the permanent, versioned project |

### 5.4 The DAG Code, Explained in Plain Terms

```python
from datetime import datetime
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

default_args = {
    "owner": "lakshmi",
    "retries": 1,
}

with DAG(
    dag_id="ai_news_pipeline",
    default_args=default_args,
    description="Daily pipeline: fetch news -> clean -> embed & store",
    schedule="@daily",
    start_date=datetime(2026, 7, 28),
    catchup=False,
    tags=["rag", "news", "ai"],
) as dag:

    fetch_task = BashOperator(
        task_id="fetch_news",
        bash_command="cd /workspaces/DE-to-MLE-Journey && python ingestion/fetch_news.py",
    )

    clean_task = BashOperator(
        task_id="clean_articles",
        bash_command="cd /workspaces/DE-to-MLE-Journey && python transform/clean_articles.py",
    )

    embed_task = BashOperator(
        task_id="embed_and_store",
        bash_command="cd /workspaces/DE-to-MLE-Journey && python embed/embed_and_store.py",
    )

    fetch_task >> clean_task >> embed_task
```

| Code | Plain-English meaning |
|---|---|
| `dag_id="ai_news_pipeline"` | The name of this whole automated job, like naming a recipe |
| `retries: 1` | If a step fails (e.g., the news website is briefly down), try it one more time automatically before giving up and alerting us — makes the system a bit more resilient to temporary hiccups |
| `schedule="@daily"` | Run this whole thing once every day, automatically, no human needed |
| `start_date` / `catchup=False` | Tells Airflow "start counting from today, and don't try to run for all the days in the past you weren't watching" — avoids it trying to "catch up" on days before we built it |
| `BashOperator` | Each task here just means "go run this existing script exactly like we did by hand" — we didn't need to rewrite our scripts, we just told Airflow to call them for us |
| `fetch_task >> clean_task >> embed_task` | This arrow syntax is the **dependency chain** — it literally means "run fetch first, then clean, then embed, in that exact order, and don't start the next one until the previous one succeeds" |

**The single most important concept to explain to someone:** *"The `>>` arrows are what make this a real pipeline instead of three separate scripts. They guarantee the data is always fetched before it's cleaned, and always cleaned before it's embedded — so we never accidentally process stale or broken data out of order."*

### 5.5 Problems We Ran Into Today (good to remember and mention in interviews — debugging is a real skill)

| Problem | What it meant | How we fixed it |
|---|---|---|
| `airflow db init` command not found | We had a newer version of Airflow (3.x) where commands got renamed | Used `airflow db migrate` instead — same purpose, new name |
| `airflow users create` command not found | Same version-naming issue — Airflow 3.x replaced manual user creation with `airflow standalone`, which auto-generates a login | Used `airflow standalone`, then retrieved the auto-generated password from a saved file |
| DAG didn't show up in the dashboard at first | Airflow was still looking in its own default folder, not our project's `dags/` folder — a classic **environment configuration mismatch** | Set an environment variable (`AIRFLOW__CORE__DAGS_FOLDER`) pointing to the correct folder, then **fully restarted** Airflow so the change actually took effect (a partial restart in a terminal that already had the old settings loaded doesn't count) |
| Deprecation warning on `BashOperator` import | The library reorganized where this tool lives internally, in a newer version | Updated the import line to the new correct location — small fix, no functional change |

**Why documenting these matters:** In a real job, things rarely work perfectly on the first try. Being able to calmly explain "here's what broke, here's why, here's how I fixed it" is often more valuable to an interviewer than a system that worked flawlessly with no story behind it.

---

## 6. Day 3 Addition: Exposing the System as a Real API (FastAPI)
**Date built:** July 29, 2026

### 6.1 The Business Problem This Solves

Even with Airflow automating the data refresh, the actual "ask a question, get an answer" part still only worked one way: someone typing directly into a terminal on this specific machine (`python api/query.py`).

**In plain, layman's terms:** imagine you built a great in-house AI assistant, but the *only* way to talk to it was to personally SSH into a server and type into a black command-line window. No website, no app, no way for a colleague, a phone app, or another piece of software to use it. That's not a usable product — it's a personal tool.

**What we solved today:** we exposed the exact same "ask a question, get an answer" logic as a **web service** — a REST API. Now, anything that can send a request over the internet (a website, a mobile app, a Slack bot, another company's software, a teammate testing it in a browser) can ask it a question and get back a clean, structured answer. This is the actual mechanism by which AI features get built into real products.

### 6.2 What FastAPI Actually Is (Layman's Explanation)

FastAPI is a tool for turning a Python function into something the internet can talk to. Think of it like putting a **counter window** in front of your pipeline: instead of someone having to walk into the back office and run the machine themselves, they slide a request through the window ("what's happening with AI regulation?") and get a response slid back ("here's the answer, and here's where it came from") — without ever needing to know or touch what's happening inside.

The two "windows" (called **endpoints**) we built:
- `GET /` — a simple "are you alive?" check, used by monitoring tools to confirm the service is running
- `POST /query` — the real one: send it a question, get back an answer + sources

### 6.3 Step-by-Step: What We Did Today

| Step | What we did | Why |
|---|---|---|
| 1 | Added `fastapi` and `uvicorn` to `requirements.txt`, installed them | FastAPI is the tool that defines the API; Uvicorn is the actual server that runs it and listens for incoming requests |
| 2 | Created `api/main.py`, reusing the same retrieve/generate logic from `query.py` | No need to reinvent the AI logic — we just changed *how* it's triggered: from "typed into a terminal" to "sent over the web" |
| 3 | Defined a `QueryRequest` structure (`question`, `n_results`) | This tells FastAPI exactly what shape of data to expect from anyone calling the API — like a form with required fields, so bad/malformed requests get rejected automatically with a clear error instead of crashing the system |
| 4 | Started the server with `uvicorn api.main:app --reload --host 0.0.0.0 --port 8000` | This actually turns the code "on" and makes it listen for incoming web requests on port 8000 |
| 5 | Tested the health check (`GET /`) | Confirmed the service was up and responding at all, before testing the real logic |
| 6 | Opened the auto-generated interactive docs (`/docs`) | FastAPI automatically builds a webpage where you can test every endpoint by filling in a form — no coding needed to try it |
| 7 | Sent a real test question through the `/query` endpoint via that docs page | Got back a proper structured JSON response: the answer, plus the exact source articles used — proof the whole chain (web request → retrieval → AI generation → response) works end-to-end |
| 8 | Committed `api/main.py` and the updated `requirements.txt` to GitHub | Saved this as a permanent, versioned part of the project |

### 6.4 The Code, Explained in Plain Terms

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI Market Intelligence API")

class QueryRequest(BaseModel):
    question: str
    n_results: int = 3

@app.get("/")
def health_check():
    return {"status": "ok", "service": "AI Market Intelligence API"}

@app.post("/query")
def query_endpoint(request: QueryRequest):
    retrieved = retrieve(request.question, request.n_results)
    answer = generate_answer(request.question, retrieved)
    sources = [
        {"title": meta["title"], "source": meta["source"], "url": meta["url"]}
        for meta in retrieved["metadatas"][0]
    ]
    return {"question": request.question, "answer": answer, "sources": sources}
```

| Code | Plain-English meaning |
|---|---|
| `app = FastAPI(...)` | Creates "the counter window" — the actual web application object everything else attaches to |
| `class QueryRequest(BaseModel)` | Defines the **required shape** of any incoming request — a question (text) is required, `n_results` is optional and defaults to 3 if not given. If someone sends something malformed, FastAPI rejects it automatically with a helpful error, before our code even runs |
| `@app.get("/")` | A **decorator** — this is Python's way of saying "when someone visits this address with a GET request, run the function right below it" |
| `def health_check()` | A simple function that just confirms the service is alive — used by monitoring systems to check "is this still running?" without doing any real work |
| `@app.post("/query")` | Same idea, but for the real endpoint — "POST" means the caller is *sending data* (their question), not just visiting a page |
| `def query_endpoint(request: QueryRequest)` | FastAPI automatically takes the incoming request, validates it matches our `QueryRequest` shape, and hands it to this function as a ready-to-use Python object |
| The body of the function | Identical logic to `query.py` from Day 1 — retrieve relevant articles, generate a grounded answer — just returned as structured JSON instead of printed to a terminal |

**The single most important concept to explain to someone:** *"This didn't require rewriting the AI logic at all — we took the exact same retrieve-and-generate code and just changed its 'front door' from a terminal prompt to a web address. That's the whole idea of an API: separating what the system does from how people access it."*

### 6.5 What We Confirmed Working

Sent a real question ("What is happening with AI regulation?") through the live API and got back a clean `200 OK` response with a grounded answer and three real, current, correctly sourced articles — proving the entire chain works exactly the same way through the API as it did through the terminal script, just now accessible over the web.

---

## 7. What's Next (not built yet)

- **Deployment (Docker + AWS):** containerize this API and host it somewhere accessible beyond this Codespace
- **Documentation polish:** architecture diagram, README for the repo itself

---

## 8. One-Sentence Summary (for a resume bullet or quick verbal explanation)

*"Built and orchestrated an end-to-end RAG pipeline that automatically ingests live news via API on a daily schedule (Apache Airflow), cleans and validates the data, generates vector embeddings for semantic search, and exposes grounded, source-cited natural-language Q&A through a FastAPI REST service — demonstrating data engineering fundamentals, workflow orchestration, and applied AI system design end-to-end."*
## Day 4 Addition: Containerizing and Deploying to AWS

### The business problem this solves

Up to this point, the whole pipeline only ran inside a personal development environment (a GitHub Codespace) — meaning nobody else could reach it, and it would disappear the moment that Codespace was deleted or timed out. Real products don't work that way: they run on servers that are always on, reachable from anywhere, and independent of any one developer's machine.

This stage answers the question: **"How do I take something that works on my machine and make it a real, live, publicly usable service?"** That's the difference between a personal script and an actual product.

### What Docker is, in plain terms

Docker packages an application together with everything it needs to run (the exact Python version, all libraries, system dependencies) into a single, portable unit called an **image**. A running copy of that image is called a **container**. The value: "it works on my machine" stops being a problem, because the container carries its own environment with it — the same image runs identically on a laptop, a Codespace, or a cloud server.

### What AWS EC2 is, in plain terms

EC2 (Elastic Compute Cloud) is Amazon's rentable virtual server. Instead of buying and maintaining physical hardware, you rent a small remote computer by the hour (free, within limits, on the free tier) that stays running and reachable on the internet — this is what makes the API "live" instead of just running temporarily on a personal machine.

### Step-by-step: what we did today

1. **Wrote a `Dockerfile`** — a recipe telling Docker how to build the image: start from a minimal Python base, install dependencies, copy in the application code, and specify the command that starts the FastAPI server.
2. **Added a `.dockerignore`** — tells Docker to skip copying things like the `data/` folder, `.env`, and `.git` into the image itself, keeping it lean and keeping secrets out of the image.
3. **Built and tested the image locally** (in the Codespace first) — hit a bug where the container crashed because the vector database (`data/`) had been deliberately excluded from the image; fixed by mounting `data/` as an external **volume** at runtime instead of baking it in — a standard practice, since application code and data should be able to change independently.
4. **Launched an AWS EC2 instance** — a small (`t2.micro`/`t3.micro`, free-tier) Ubuntu server, with a **security group** controlling exactly which network traffic is allowed in (SSH on port 22 for admin access, port 8000 for the API itself).
5. **Connected to the instance via SSH**, installed Docker on it directly, cloned the GitHub repo onto the server, and rebuilt the image there (a real image needs to be built on the machine that will run it, or transferred to it — we chose to rebuild).
6. **Ran the ingestion → clean → embed pipeline directly on the EC2 server** to populate a fresh vector database there, since the original one only existed in the Codespace and wasn't tracked in git (intentionally — see the `.gitignore` decision from Day 1).
7. **Ran the Docker container on EC2** with the data volume mounted and the API keys passed in via `.env`, exposing it on port 8000.
8. **Verified public reachability** by hitting `http://<public-IP>:8000/` directly from a browser (not the server itself) — confirming the API is now live on the open internet, not just accessible locally.

### Real problems hit today, and how they were fixed

| Problem | Root cause | Fix |
|---|---|---|
| SSH / EC2 Instance Connect kept failing ("Connection timed out") | Security group's SSH rule only allowed one specific IP, and both the browser-based Instance Connect tool and the Codespace's SSH client come from different IPs than expected | Widened the security group's SSH and port-8000 rules to allow broader access (acceptable for a temporary demo instance) |
| `docker build` failed with a networking error on `docker run` | The `docker.io` Ubuntu package shipped without a required proxy binary in the expected place | Set `"userland-proxy": false` in Docker's daemon config |
| Pasting into the browser-based SSH terminal didn't work | Browser terminal doesn't support standard clipboard shortcuts | Switched to SSHing into the instance from the Codespace terminal instead, where paste worked normally |
| `docker build` ran out of disk space | The default install pulled in the full GPU/CUDA version of PyTorch (multiple GB), unnecessary on a CPU-only server | Modified the Dockerfile to explicitly install the CPU-only PyTorch build first, avoiding the CUDA packages entirely |
| Python package install got silently `Killed` mid-download | The EC2 instance only has ~900MB of RAM with no swap space, so a large install got killed by Linux's out-of-memory protection | Added a swap file to give the instance breathing room during heavy installs |
| `pip install` failed with "disk quota exceeded" | pip's temp directory was on a constrained mount | Redirected pip's temp directory (`TMPDIR`) to a location with more space |
| Container kept crashing on startup (`Collection [news_articles] does not exist`) | The vector database only existed in the original Codespace; a fresh `git clone` on EC2 had no data (correctly, since `data/` is gitignored) | Re-ran the ingestion → clean → embed pipeline directly on the EC2 server to build a fresh vector database there |
| Ran out of disk space again during rebuild, even after cleanup | The 8GB root volume was fundamentally too small for the OS + Docker + ML dependencies together | Resized the EBS volume from 8GB to 20GB in the AWS console, then grew the Linux partition and filesystem to use the new space |
| A script wrote a file and got `PermissionError` | Folder ownership mismatch, likely left over from an earlier root-owned Docker write | Reset ownership of the project folder to the current user with `chown -R` |

### What this demonstrates (interview-ready summary)

*"I containerized a RAG pipeline with Docker and deployed it to a live AWS EC2 instance, debugging real infrastructure constraints along the way — memory limits, disk sizing, dependency bloat from unnecessary GPU packages, and network/security group configuration. The result is a publicly reachable REST API backed by an automated, scheduled data pipeline — the same overall shape as a real production AI feature."*

### What's next
- Add a domain name + HTTPS in front of the API (currently accessed via raw IP and HTTP)
- Consider basic API-key auth before sharing the endpoint more broadly
- Optionally move to a managed container service (ECS/Fargate) instead of a single EC2 instance for better reliability