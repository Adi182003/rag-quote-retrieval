from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
import os

ds = load_dataset("Abirate/english_quotes")
df = pd.DataFrame(ds['train'])
df = df.dropna(subset=['quote', 'author', 'tags'])
df['quote'] = df['quote'].str.strip().str.lower()

model = SentenceTransformer('all-MiniLM-L6-v2')
train_examples = [InputExample(texts=[row['quote'], row['quote']], label=1.0)
                  for i, row in df.sample(1000, random_state=42).iterrows()]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune and save
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=10)
model.save("fine_tuned_model")