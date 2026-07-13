import re
import streamlit as st
from huggingface_hub import InferenceClient

# Ensure persistent client storage
# if "hf_client" not in st.session_state:
#     st.session_state["hf_client"] = None


def init_model(token):
    """
    Initialize the Llama model once and store inside Streamlit session_state.
    """
    if "hf_client" not in st.session_state:
        st.session_state.hf_client = InferenceClient( token=token, timeout=15)
    print("HF client initialized.")

def is_bad_response(text):
    """
    Validates the LLM output to catch silent failures, truncations, or language loops.
    """
    if text is None or len(text.strip()) < 20:
        return True
    
    text_lower = text.lower()
    
    # 1. Check for standard empty/refusal patterns
    refusals = [
        "i cannot answer", "i am sorry", "i'm sorry", 
        "unable to provide", "timeout", "internal server error"
    ]
    if any(refusal in text_lower for refusal in refusals):
        return True
        
    # 2. Check for repetitive token patterns (e.g., a word repeated 3+ times in a row)
    # This acts as an automated guardrail for low-resource script looping bugs
    if re.search(r'\b(\w+)(?:\s+\1\b){3,}', text_lower):
        return True
        
    return False

def generate_with_context(query, context):
    """
    Generate answer grounded in context using chat_completion.
    """
    if "hf_client" not in st.session_state:
        raise RuntimeError("HF client not initialized. Call init_model() first.")
    
    client = st.session_state.hf_client

    if not context:
        return "I am sorry, but that topic falls outside the scope of the Occupational Safety, Health and Working Conditions Code, 2020."

    messages = [
        {"role": "system", "content": (
            "You are an expert Occupational Safety and Health (OSH) assistant. "
            "You will be given English legal texts as context. Answer the user's query accurately using ONLY this context. "
            "CRITICAL DIRECTIVE: Identify the language of the USER QUESTION. You MUST write your entire response "
            "in that exact same language. Do not mix multiple languages. Do not switch languages mid-response. "
            "Provide the answer directly without any conversational filler or internal reasoning."
        )},
        {"role": "user", "content": (
            f"--- ENGLISH CONTEXT ---\n{context}\n-----------------------\n\n"
            f"USER QUESTION: {query}\n\n"
            "REMINDER: Generate the final answer exclusively in the language of the USER QUESTION. Do not use any other language."
        )}
    ]

    try:
        print("Attempting generation with Primary Model: Qwen 2.5...")
        response = client.chat_completion(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=messages,
            max_tokens=800,           
            temperature=0.2,
            top_p=0.9,
            frequency_penalty=0.5,    
            # repetition_penalty=1.1
        )
        
        answer = response.choices[0].message["content"]

    except Exception as e:
        # Intercepts 500, 503, connection drops, or timeout exceptions cleanly
        print(f"❌ Primary API Error (Qwen Failed): {e}")
        
    # Verify the quality of Qwen's response before sending it to the front-end
    if answer is not None:
        if not is_bad_response(answer):
            return answer
        else:
            print("⚠️ Qwen response failed validation check. Triggering failover...")
            
    


    # --- FALLBACK ATTEMPT: LLAMA 3 ---
    try:
        print("🔄 Executing Fallback Strategy: Routing request to Llama 3...")
        response = client.chat_completion(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=messages,
            max_tokens=800,           
            temperature=0.35,  # Slightly relaxed for fallback vocabulary variations
            top_p=0.9,
            frequency_penalty=0.5,
        )
        return response.choices[0].message["content"]
        
    except Exception as e:
        print(f"❌ Fallback API Error (Llama 3 Failed): {e}")
        return "The AI system is experiencing brief technical issues or high service load. Please wait a moment and resubmit your query."

    # response = client.chat_completion(
    #     model=model_id,
    #     messages=messages,
    #     max_tokens=800,
    #     temperature=0.2,          # Increased from 0.1 to prevent the repetition loop in Bengali
    #     top_p=0.9,
    #     frequency_penalty=0.5,     # Actively forces the model to stop repeating the same words
    #     repetition_penalty=1.1
    # )

    # return response.choices[0].message["content"]
