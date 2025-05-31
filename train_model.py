import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["ACCELERATE_DISABLE_MPS"] = "1"
os.environ["TRANSFORMERS_NO_MPS"] = "1"

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, SentencesDataset

print("✅ Forcing CPU only")

def fine_tune_model():
    # Load dataset
    df = pd.read_csv('data/english_quotes.csv')

    # Clean quotes
    df['quote'] = df['quote'].str.strip().str.replace(r'[^\w\s]', '', regex=True)
    df = df[df['quote'].str.split().str.len() > 5]

    # Group by author and create training pairs
    author_groups = df.groupby('author')['quote'].apply(list).to_dict()
    train_samples = []

    for quotes in author_groups.values():
        if len(quotes) >= 2:
            for i in range(len(quotes)):
                for j in range(i + 1, len(quotes)):
                    train_samples.append(InputExample(texts=[quotes[i], quotes[j]]))

    print(f"🔢 Total training pairs: {len(train_samples)}")

    # Load SentenceTransformer model on CPU
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    # Create dataloader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=8)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    print("🚀 Training on CPU...")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=10,
        output_path='models/quote_bert_model',
        show_progress_bar=True
    )

    print("🎉 Training completed and model saved to models/quote_bert_model")

if __name__ == "__main__":
    fine_tune_model()
