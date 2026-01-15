from flask import Flask, request, jsonify, render_template
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF INFO / WARN

import tensorflow as tf
# --------------------------------
# Load PDF
# --------------------------------
reader = PdfReader("data.pdf")
text = " ".join(page.extract_text() or "" for page in reader.pages)

chunks = [text[i:i+500] for i in range(0, len(text), 500)]

# --------------------------------
# Load models
# --------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# --------------------------------
# Build vector index
# --------------------------------
embeddings = embedder.encode(chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# --------------------------------
# Flask app
# --------------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

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

    return render_template("index.html", answer=answer, question=question)

if __name__ == "__main__":
    app.run(port=8000)
