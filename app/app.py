import os
import sys
import ast
import re
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import time
from numpy.linalg import norm


USER_AVATAR = "https://cdn-icons-png.flaticon.com/512/847/847969.png"
BOT_AVATAR  = "https://cdn-icons-png.flaticon.com/512/4712/4712100.png"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import init_model, generate_with_context
from src.retriever import retrieve_top_sections, build_faiss_index

load_dotenv()
if "hf_client" not in st.session_state:
    try:
        token = st.secrets["HF_TOKEN"]
    except Exception:
        token= os.getenv("HF_TOKEN")  # local fallback only

    if not token:
        st.error("HF_TOKEN not found. Please configure Streamlit secrets.")
        st.stop()

    init_model(token)

# ----------------------------------
# Helper to extract Section numbers
# ----------------------------------
def extract_section_number(text):
    """
    Detects section number from OSH section text.
    Handles patterns like:
    '12. Duties of employer'
    '23. Notice of certain diseases'
    '3. (1) ...'
    """
    # Look for pattern like "23." or "12."
    match = re.search(r"^\s*(\d+)\.", text.strip())
    if match:
        sec_num = match.group(1)
        return f"Section {sec_num}"
    
    return "Section ?"



# ----------------------------------
# Initialize all components (cached)
# ----------------------------------

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


@st.cache_resource
def init_all():
    CSV_PATH = "data/processed/osh_sections_with_vectors.csv"
    df = pd.read_csv(CSV_PATH)

    # Parse vector embeddings if stored as string
    df["vector_embedding"] = df["vector_embedding"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Build FAISS index
    vecs = np.array(df["vector_embedding"].tolist()).astype("float32")
    index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)

    corpus_vectors = np.array(df["vector_embedding"].tolist())
    corpus_centroid = corpus_vectors.mean(axis=0)


    return df, embedder, index, corpus_centroid


df, embedder, index, corpus_centroid = init_all()


# ----------------------------------
# Retrieval + Generation with Citations
# ----------------------------------

def is_refusal(answer: str) -> bool:
    refusal_patterns = [
        r"^i can[’']?t provide information on",
        r"^i cannot provide information on",
        r"^this question is outside",
        r"^i am unable to help with",
        r"^this is not covered under",
        r"^please provide the term you would like to have defined",
    ]
    answer = answer.strip().lower()
    return any(re.match(pat, answer) for pat in refusal_patterns)


def is_osh_related(query, embedder, corpus_centroid, threshold=0.30):
    q_vec = embedder.encode(query)
    sim = cosine_similarity(q_vec, corpus_centroid)
    return sim >= threshold

def is_greeting(query):
    return bool(re.match(
        r"^(hi|hello|hey|good morning|good evening)\b",
        query.strip().lower()
    ))

def is_identity_query(query: str) -> bool:
    return bool(re.match(
        r"^(who\s+are\s+you|what\s+is\s+your\s+name|what\s+are\s+you|introduce\s+yourself)\b",
        query.strip().lower()
    ))

OSCAR_IDENTITY_RESPONSE = (
    "I am 𝐎.𝐒.𝐂.𝐀.𝐑. (Occupational Safety Compliance & Regulation) Chatbot, trained to answer questions based on the Occupational Safety, Health and Working Conditions Code, 2020. How may I assist you today?"
)

def answer_with_citations(query, top_k=3):
    if is_greeting(query) or is_identity_query(query):
        return OSCAR_IDENTITY_RESPONSE, [], []
    
    if not is_osh_related(query, embedder, corpus_centroid):
        answer = generate_with_context(query, context="")
        return answer, [], []
    
    answer = generate_with_context(query, context="")

    if is_refusal(answer):
        return answer, [], []
    
    top_sections = retrieve_top_sections(
        query, embedder, df, index, k=top_k
    )

    citations = []
    for section_text in top_sections:
        sec = extract_section_number(section_text)
        citations.append(sec)

    context = "\n\n---\n\n".join(top_sections)

    answer = generate_with_context(query, context)

    return answer, top_sections, citations



# ----------------------------------
# PAGE SETTINGS
# ----------------------------------
st.set_page_config(
    page_title="OSH Compliance Chatbot",
    page_icon="🦺",
    layout="centered"
)

col1, col2 = st.columns([5, 1])   # Wider left, narrow right

# ----------------------------------
# TITLE
# ----------------------------------
with col1:
    st.markdown("""
        <h1 style="margin-bottom: 0; font-size:40px">🦺 O.S.C.A.R.</h1>
        <p style="font-size:16px; margin-top:0; margin-bottom:30px;">
            Occupational Safety Compliance & Regulation <br> 
            AI Assistant for the Occupational Safety, Health & Working Conditions Code, 2020
        </p>
    """, unsafe_allow_html=True)

# CLEAR CHAT BUTTON
with col2:
    st.markdown(
        """
        <div style="height: 30px;"></div>  """,
        unsafe_allow_html=True
    )
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ----------------------------------
# CHAT HISTORY
# ----------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add the flag here:
if "generating_response" not in st.session_state:
    st.session_state.generating_response = False

# ----------------------------------
# DISPLAY MESSAGES
# ----------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-end; margin-bottom:20px;">
                <div style="
                    background-color:#2F2F2F;
                    padding:12px;
                    border-radius:10px;
                    max-width:70%;
                    text-align:right;
                ">
                    {msg['content']}
                </div>
                <img src="{USER_AVATAR}" width="40" style="margin-left:10px; border-radius:50%;">
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f"""
            <div style="display:flex; align-items:flex-start; margin-bottom:20px;">
                <img src="{BOT_AVATAR}" width="40" style="margin-right:15px; margin-top:5px; border-radius:50%; align-self:flex-start;">
                <div style="
                    background-color:#1E1E1E;
                    padding:16px;
                    border-radius:10px;
                    max-width:80%;
                ">
                    {msg['content']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
        # Show citations
        if "citations" in msg:
            st.markdown(f"🔍 **Cited Sections:** {', '.join(msg['citations'])}")

        # Expandable context viewer
        if "context" in msg:
            with st.expander("📘 View retrieved context"):
                for i, sec in enumerate(msg["context"], start=1):
                    st.markdown(f"**Context {i}:**\n\n{sec}")


# ----------------------------------
# INPUT HANDLER CALLBACK
# ----------------------------------
def submit():
    st.session_state["pending_user_msg"] = st.session_state["new_input"]
    st.session_state["new_input"] = ""


# ----------------------------------
# INPUT BOX
# ----------------------------------
if not st.session_state.generating_response:
    st.text_input(
        "Ask your OSH Code question:",
        placeholder="Type your question...",
        key="new_input",
        on_change=submit
    )

# ----------------------------------
# PROCESS USER QUERY
# ----------------------------------
if "pending_user_msg" in st.session_state and st.session_state["pending_user_msg"]:
    user_input = st.session_state["pending_user_msg"]

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state["pending_user_msg"] = "" 
    st.session_state.generating_response = True  

    st.rerun()

if st.session_state.generating_response:
    user_input = st.session_state.messages[-1]["content"]

    with st.spinner("Thinking..."):
        answer, context_list, citations = answer_with_citations(user_input)

    placeholder = st.empty()
    streamed_text = ""
    typing_delay = 0.03

    # --- NEW STREAMING LOOP ---
    for char in answer:
        streamed_text += char

        if char.isspace() or char in '.,:;?!':
            
            placeholder.markdown(
                f"""
                <div style="display:flex; align-items:flex-start; margin-bottom:20px;">
                    <img src="{BOT_AVATAR}" width="40" style="margin-right:15px; margin-top:5px; border-radius:50%; align-self:flex-start;">
                    <div style="
                        background-color:#1E1E1E;
                        padding:16px;
                        border-radius:10px;
                        max-width:80%;
                    ">
                        {streamed_text}<span style="opacity:0.8;">▌</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            time.sleep(typing_delay)

    placeholder.empty()

    msg = {
    "role": "assistant",
    "content": answer
    }
    if context_list:
        msg["context"] = context_list
    if citations:
        msg["citations"] = citations

    st.session_state.messages.append(msg)

    st.session_state.generating_response = False

    st.rerun()

# ----------------------------------
# FOOTER
# ----------------------------------
st.markdown("""
<hr>
<div style="text-align:center; color:#777;">
    Built using RAG (FAISS + MiniLM + Llama-3)
</div>
""", unsafe_allow_html=True)
