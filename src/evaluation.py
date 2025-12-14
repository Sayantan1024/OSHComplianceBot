import os
import sys
import ast
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()

# Add parent directory to sys.path to ensure module imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import your RAG pipeline functions (Assuming these paths are correct)
# Note: build_faiss_index is expected to be defined in src.retriever
from src.retriever import retrieve_top_sections, build_faiss_index
from src.inference import init_model, generate_with_context


# -----------------------------
# LOAD MODELS & DATA
# -----------------------------
def load_all():
    """Loads datasets, models, and initializes RAG components."""
    print("Initializing RAG components...")
    
    # 1. Load evaluation dataset
    EVAL_CSV_PATH = "data/processed/evaluation_questions.csv"
    try:
        eval_df = pd.read_csv(EVAL_CSV_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"Evaluation CSV not found at: {EVAL_CSV_PATH}")

    # 2. Load OSH dataset with vectors
    OSH_CSV_PATH = "data/processed/osh_sections_with_vectors.csv"
    try:
        df = pd.read_csv(OSH_CSV_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"OSH sections CSV not found at: {OSH_CSV_PATH}")
    
    # Convert embedding strings back to lists
    df["vector_embedding"] = df["vector_embedding"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # 3. Load embedding model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # 4. Build FAISS index
    # Assumes build_faiss_index can handle the dataframe 'df'
    index = build_faiss_index(df) 

    # 5. Initialize LLM
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable not set.")
    init_model(HF_TOKEN)

    print("Initialization complete.")
    return eval_df, df, embedder, index


# -----------------------------
# RAG Answer Generator for Evaluation
# -----------------------------
def answer_query_eval(query, df, embedder, index, k=2):
    """Generates an answer using the RAG pipeline."""
    top_sections = retrieve_top_sections(query, embedder, df, index, k=k)
    context = "\n\n---\n\n".join(top_sections)
    # Note: Assumes generate_with_context handles the prompt and LLM call
    answer = generate_with_context(query, context) 
    return answer


# -----------------------------
# SEMANTIC SIMILARITY CHECK (Efficient)
# -----------------------------
def is_correct(pred, gold, embedder, threshold=0.75):
    """
    Compares the predicted answer to the expected answer using cosine similarity.
    Returns 1 if score >= threshold (Correct), 0 otherwise (Incorrect).
    """
    # The embedder is passed in to avoid reloading the model in a loop
    emb1 = embedder.encode(pred, convert_to_tensor=True)
    emb2 = embedder.encode(gold, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    return 1 if score >= threshold else 0


# -----------------------------
# MAIN EVALUATION LOOP
# -----------------------------
def run_evaluation():
    eval_df, df, embedder, index = load_all()

    predicted_answers = []

    print("\n🔍 Running RAG evaluation… This may take a few minutes.\n")
    for q in tqdm(eval_df["question"]):
        ans = answer_query_eval(q, df, embedder, index)
        predicted_answers.append(ans)

    eval_df["predicted_answer"] = predicted_answers
    
    # TRUE LABEL: All samples in the evaluation set are assumed to be correct (1)
    eval_df["true_label"] = 1  
    
    # PREDICTED LABEL: Check semantic similarity against the expected (gold) answer
    eval_df["pred_label"] = [
        is_correct(pred, gold, embedder)
        for pred, gold in zip(eval_df["predicted_answer"], eval_df["expected_answer"])
    ]

    # --- Prepare for Metrics ---
    y_true = eval_df["true_label"].values
    y_pred = eval_df["pred_label"].values
    
    # -----------------------------
    # CONFUSION MATRIX
    # -----------------------------
    # The default labels=[0, 1] might fail if y_true only contains 1s.
    # We use a try/except block to ensure a 2x2 matrix is always produced.
    # -----------------------------
    # CONFUSION MATRIX CALCULATION AND PLOTTING
    # -----------------------------
    
    # Define the class labels used for the evaluation
    class_labels = ["Incorrect (0)", "Correct (1)"]
    
    # Calculate the confusion matrix. 
    # The try/except is a good defensive measure for when the 'Incorrect' class (0) is missing.
    try:
        # Use explicit labels=[0, 1] for a guaranteed 2x2 matrix structure
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    except ValueError:
        # Fallback if y_true only contains '1's (Actual Correct)
        fn_count = np.sum((y_true == 1) & (y_pred == 0)) # False Negatives
        tp_count = np.sum((y_true == 1) & (y_pred == 1)) # True Positives
        cm = np.array([[0, 0], [fn_count, tp_count]]) # [[TN, FP], [FN, TP]]

    # Create a DataFrame for cleaner visualization labels
    cm_df = pd.DataFrame(cm, 
                         index = [f'Actual {label}' for label in class_labels],
                         columns = [f'Predicted {label}' for label in class_labels])
    
    # Plotting the Heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",  # Format as integer (d) counts
        cmap="Blues",
        cbar=True,
        linewidths=0.5, # Optional: Adds lines between cells
        linecolor='gray'
    )
    plt.title("Confusion Matrix — RAG Evaluation Performance")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    # Save plot
    os.makedirs("data/output", exist_ok=True)
    plt.savefig("data/output/confusion_matrix.png", dpi=300)
    print("\n📊 Confusion matrix saved as: data/output/confusion_matrix.png")

    # -----------------------------
    # CLASSIFICATION REPORT
    # -----------------------------
    print("\n📄 Classification Report:")
    # Use zero_division='warn' for cases where precision/recall are undefined (e.g., if no 0s are predicted)
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))

    # Save evaluation results
    eval_df.to_csv("data/output/evaluation_results.csv", index=False)
    print("\n💾 Evaluation results saved to: data/output/evaluation_results.csv\n")


# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    run_evaluation()