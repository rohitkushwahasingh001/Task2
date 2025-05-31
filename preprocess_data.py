# preprocess_data.py
from datasets import load_dataset
import pandas as pd
import os 

def download_and_preprocess():
    print("Downloading dataset...")
    dataset = load_dataset("Abirate/english_quotes")
    df = pd.DataFrame(dataset['train'])
    
    # Clean text
    df['quote'] = df['quote'].str.strip()
    df['quote'] = df['quote'].str.replace(r'[^\w\s\.\,\!\?\"]', '', regex=True)
    df['quote'] = df['quote'].str.replace(r'\s+', ' ', regex=True)
    df['quote'] = df['quote'].str.lower()

    # Remove very short quotes
    df = df[df['quote'].str.split().str.len() > 5]

    print(f"Downloaded {len(df)} quotes.")
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/english_quotes.csv', index=False)
    return df

if __name__ == "__main__":
    download_and_preprocess()