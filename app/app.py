# app/app.py

import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from src.inference import init_model, generate_with_context
from src.retriever import build_faiss_index, retrieve_top_sections

# ==========================
# 🔧 Load environment variables
# ==========================
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found in .env file.")

# ==========================
# 🤖 Initialize Llama model
# ==========================
init_model(HF_TOKEN)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "osh_sections_with_vectors.csv")

# ==========================
# 📄 Load processed OSH data
# ==========================
df = pd.read_csv(DATA_PATH)

# Convert embeddings from string → list if needed
df["vector_embedding"] = df["vector_embedding"].apply(eval)

# ==========================
# 🔍 Build FAISS Index
# ==========================
index = build_faiss_index(df)

# ==========================
# 🧠 Load embedding model
# ==========================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================
# 🦺 RAG pipeline function
# ==========================
def answer_query(query, k=3):
    top_sections = retrieve_top_sections(query, embedder, df, index, k)
    context = "\n\n---\n\n".join(top_sections)
    answer = generate_with_context(query, context)
    return answer

# ==========================
# 💬 CLI Chat Interface
# ==========================
def start_chat():
    print("\n🦺 Welcome to the AI-Powered OSH Compliance Chatbot!")
    print("Ask me anything about the Occupational Safety, Health and Working Conditions Code, 2020.")
    print("Type 'exit' to end the chat.\n")

    while True:
        query = input("👤 You: ").strip()
        if query.lower() in ["exit", "quit", "stop"]:
            print("👋 Goodbye! Stay safe and compliant.")
            break

        response = answer_query(query)
        print(f"\n🤖 Bot: {response}\n")

if __name__ == "__main__":
    start_chat()
