import pandas as pd
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import pickle

ds = load_dataset("Abirate/english_quotes")
df = pd.DataFrame(ds['train'])
df = df.dropna(subset=['quote', 'author', 'tags'])
df['quote'] = df['quote'].str.strip().str.lower()
df['author'] = df['author'].str.strip().str.lower()
df['tags'] = df['tags'].apply(lambda tags: [t.lower() for t in tags] if isinstance(tags, list) else [])

model = SentenceTransformer('fine_tuned_model')  # Load your fine-tuned model

embeddings = model.encode(df['quote'].tolist(), show_progress_bar=True, convert_to_numpy=True)
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

# Save index and dataframe
faiss.write_index(faiss_index, "quotes.index")
df.to_pickle("quotes_df.pkl")