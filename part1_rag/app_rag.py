import os
import json
import numpy as np
import sqlite3
import streamlit as st
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_API_TOKEN")

DATA_DIR = "part1_rag/docs"
STORE_DIR = "rag_store"
EMBED_MODEL = "all-MiniLM-L6-v2"
# HF_MODEL = "HuggingFaceH4/zephyr-7b-beta"
HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


os.makedirs(STORE_DIR, exist_ok=True)

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def is_injection(q):
    bad = ["secret", "reveal", "dump", "password", "show files"]
    return any(b in q.lower() for b in bad)


def load_docs():
    docs = []
    for fn in os.listdir(DATA_DIR):
        if fn.endswith(".md"):
            txt = open(os.path.join(DATA_DIR, fn), encoding="utf-8").read()
            docs.append({"id": fn, "text": txt})
    return docs

@st.cache_resource
def embedder():
    return SentenceTransformer(EMBED_MODEL)

def build_json(docs, model):
    path = f"{STORE_DIR}/vectors.json"
    if os.path.exists(path):
        return json.load(open(path))
    store = []
    for d in docs:
        store.append({
            "id": d["id"],
            "text": d["text"],
            "emb": model.encode(d["text"]).tolist()
        })
    json.dump(store, open(path, "w"))
    return store


def build_sqlite(docs, model):
    path = f"{STORE_DIR}/vectors.sqlite"
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS docs (id TEXT, text TEXT, emb TEXT)")
    if c.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 0:
        for d in docs:
            emb = json.dumps(model.encode(d["text"]).tolist())
            c.execute("INSERT INTO docs VALUES (?, ?, ?)", (d["id"], d["text"], emb))
        conn.commit()
    rows = c.execute("SELECT * FROM docs").fetchall()
    return [{"id": r[0], "text": r[1], "emb": json.loads(r[2])} for r in rows]



def build_qdrant(docs, model):
    client = QdrantClient("localhost", port=6333)
    client.recreate_collection(
        collection_name="docs",
        vectors_config={"size": 384, "distance": "Cosine"}
    )
    for i, d in enumerate(docs):
        client.upsert(
            collection_name="docs",
            points=[{
                "id": i,
                "vector": model.encode(d["text"]).tolist(),
                "payload": d
            }]
        )
    return client



def retrieve(store, q, model, k, backend):
    q_emb = model.encode(q)

    if backend == "qdrant":
        hits = store.search("docs", q_emb.tolist(), limit=k)
        return [(h.score, h.payload) for h in hits]

    scored = [
        (cosine(q_emb, np.array(d["emb"])), d)
        for d in store
    ]
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:k]

def call_llm(context, question):
    messages = [
        {"role": "system", "content": "Use ONLY the context to answer. Max 100 words."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
    ]
    client = InferenceClient(api_key=HF_TOKEN)
    response = client.chat_completion(
        messages,
        model=HF_MODEL,
        max_tokens=200
    )
    
    return response.choices[0].message.content


st.title("RAG with Selectable Vector Store")

backend = st.sidebar.selectbox("Storage backend", ["json", "sqlite", "qdrant"])
k = st.sidebar.number_input("Top-k", 1, 10, 5)

question = st.text_input("Ask a question about the docs")

docs = load_docs()
model = embedder()


if backend == "json":
    store = build_json(docs, model)
elif backend == "sqlite":
    store = build_sqlite(docs, model)
else:
    store = build_qdrant(docs, model)

if question:
    if is_injection(question):
        st.error("I can’t help with that request. Please ask a safe question about the docs.")
    else:
        hits = retrieve(store, question, model, k, backend)
        context = "\n\n".join(h[1]["text"] for h in hits)

        answer = call_llm(context, question)
        citations = list({h[1]["id"] for h in hits})

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Citations")
        st.table({"doc_id": citations})

        with st.expander("Debug"):
            for score, d in hits:
                st.write(f"**{d['id']}**")
                st.write(d["text"][:200] + "...")

        print(json.dumps({
            "answer": answer,
            "citations": citations
        }, indent=2))