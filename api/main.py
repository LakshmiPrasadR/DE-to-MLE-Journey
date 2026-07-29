import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="AI Market Intelligence API")

client_ai = Groq(api_key=os.getenv("GROQ_API_KEY"))
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="data/vector_store")
collection = chroma_client.get_collection(name="news_articles")


class QueryRequest(BaseModel):
    question: str
    n_results: int = 3


def retrieve(query, n_results=3):
    query_embedding = embed_model.encode([query]).tolist()
    return collection.query(query_embeddings=query_embedding, n_results=n_results)


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