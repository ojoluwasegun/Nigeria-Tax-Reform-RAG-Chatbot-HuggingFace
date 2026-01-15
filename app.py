from flask import Flask, request, render_template
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import requests

app = Flask(__name__)

# -------------------------------
# Hugging Face API settings
# -------------------------------
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HF_TOKEN = os.environ.get("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")  # set this in Vercel or locally

def generate_answer(prompt):
    """Call Hugging Face Inference API for text generation"""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"]
    elif isinstance(data, dict) and "error" in data:
        return f"Error: {data['error']}"
    else:
        return str(data)

# -------------------------------
# Load PDF & build RAG index
# -------------------------------
pdf_path = "data/data.pdf"  # adjust path
reader = PdfReader(pdf_path)
text = " ".join(page.extract_text() or "" for page in reader.pages)

# Split text into chunks
chunks = [text[i:i+500] for i in range(0, len(text), 500)]

# Encode chunks and build FAISS index
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# -------------------------------
# Flask routes
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")  # assumes templates/index.html

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")

    # Retrieve relevant context using FAISS
    q_emb = embedder.encode([question])
    _, ids = index.search(np.array(q_emb), 2)
    context = " ".join([chunks[i] for i in ids[0]])

    # Construct prompt for Hugging Face
    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""

    # Generate answer via Hugging Face API
    answer = generate_answer(prompt)

    return render_template("index.html", answer=answer, question=question)

# -------------------------------
# Run locally
# -------------------------------
if __name__ == "__main__":
    app.run(port=8000)
