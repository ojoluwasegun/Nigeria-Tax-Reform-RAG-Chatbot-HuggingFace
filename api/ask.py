from flask import Flask, request, render_template
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import os

app = Flask(__name__)

# Load PDF
reader = PdfReader("data/data.pdf")
text = " ".join(page.extract_text() or "" for page in reader.pages)
chunks = [text[i:i+500] for i in range(0, len(text), 500)]

# Load models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# Build index
embeddings = embedder.encode(chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

@app.route("/")
def home():
    return render_template("../templates/index.html")  # adjust path

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    q_emb = embedder.encode([question])
    _, ids = index.search(np.array(q_emb), 2)
    context = " ".join([chunks[i] for i in ids[0]])

    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""
    answer = generator(prompt, max_length=150)[0]["generated_text"]

    return render_template("../templates/index.html", answer=answer, question=question)
