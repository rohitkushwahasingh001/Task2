# build_rag_pipeline.py
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np
import os

def build_vectorstore():
    print("Building FAISS index...")
    df = pd.read_csv('data/english_quotes.csv')
    model = SentenceTransformer('models/quote_bert_model')
    embeddings = model.encode(df['quote'].tolist(), convert_to_numpy=True)

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index
    os.makedirs('vectorstore', exist_ok=True)
    faiss.write_index(index, 'vectorstore/faiss_index.bin')
    print("FAISS index built and saved.")

    return index, df, model

if __name__ == "__main__":
    build_vectorstore()