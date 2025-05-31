# app.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Patch for PyTorch + Streamlit bug on macOS
import sys
if os.name != "nt":
    import nest_asyncio
    nest_asyncio.apply()

    class _DummyPath:
        _path = []

    sys.modules['torch.classes'] = _DummyPath()

# Import Streamlit and Set Page Config First
import streamlit as st
st.set_page_config(page_title="💬 RAG Quote Assistant", layout="centered")

# Rest of imports
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load resources
@st.cache_resource
def load_resources():
    model = SentenceTransformer('models/quote_bert_model')
    index = faiss.read_index('vectorstore/faiss_index.bin')
    df = pd.read_csv('data/english_quotes.csv')
    return model, index, df

model, index, df = load_resources()

# Function to call Groq model for explanation
def generate_response(prompt):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=300,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input form
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("Enter your query (e.g., 'Quotes about courage by women authors')", key="input")
    submit_button = st.form_submit_button(label="Send")

if submit_button and user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Retrieve top matches
    query_emb = model.encode([user_input])
    D, I = index.search(query_emb, k=3)
    results = df.iloc[I[0]]

    # Add assistant response with results
    st.session_state.messages.append({
        "role": "assistant",
        "content": {
            "query": user_input,
            "results": results,
            "distances": D[0]
        }
    })

# Display conversation history in chat style
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            content = msg["content"]
            st.markdown(f"🔍 Matching quotes for: **{content['query']}**")

            table_data = []
            for idx in range(len(content["results"])):  # Safely loop by index
                row = content["results"].iloc[idx]
                quote = row.get("quote", "")
                author = row.get("author", "Unknown")
                

                
              

                table_data.append({
                    "Author": author,
                    "Quote": quote,
            
                    "Similarity Score": round(1 / (1 + content["distances"][idx]), 4)
                })

            st.table(table_data)

            if st.button("🧠 Show Explanation", key=f"explain_{i}"):
                combined_prompt = (
                    f"Explain the relevance of these quotes to the query: '{content['query']}'\n\n"
                    f"1. {table_data[0]['Quote']}\n"
                    f"2. {table_data[1]['Quote']}\n"
                    f"3. {table_data[2]['Quote']}"
                )
                explanation = generate_response(combined_prompt)
                st.markdown("### 🧠 Explanation:")
                st.write(explanation)