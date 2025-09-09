import pandas as pd
import numpy as np
import re
import os
from collections import Counter
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModel
import torch

# CONFIG
DATASET_PATH = 'Capstone---1/En-Ba-Dataset(20k_4)/dataset.csv'
EMBEDDING_DIM = 100
OUTPUT_PATH = 'embedded_data/dataset_numeric_final.csv'
BINARY_OUTPUT_PATH = 'embedded_data/dataset_binary_final.csv'
VOCAB_PATH = 'vocabulary.txt'
MODEL_NAME = 'bert-base-multilingual-cased'  # multilingual BERT
BITS_PER_NUMBER = 8  # Using 8 bits for each number

def preprocess(text):
    """Preprocess text by lowercasing and removing special characters"""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def build_vocab(sentences):
    """Build vocabulary from all words"""
    vocab = Counter()
    for sent in sentences:
        tokens = preprocess(sent).split()
        vocab.update(tokens)
    return {word: {"index": idx, "count": count} 
            for idx, (word, count) in enumerate(vocab.most_common())}

def save_vocab(vocab, filepath):
    """Save vocabulary to file as a dictionary format"""
    sorted_vocab = dict(sorted(vocab.items(), key=lambda x: x[1]['count'], reverse=True))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(sorted_vocab, f, indent=2, ensure_ascii=False)
    print(f'\nTop 20 most common words:')
    for i, (word, info) in enumerate(sorted_vocab.items()):
        if i >= 20: break
        print(f'{word}: {info["count"]}')

def normalize_vector(vec):
    """Convert vector to positive integers between 1 and 255"""
    # Take absolute values
    vec = np.abs(vec)
    # Normalize to range [1, 255] to fit in 8 bits
    vec_min, vec_max = vec.min(), vec.max()
    if vec_max == vec_min:
        return np.ones_like(vec, dtype=int)
    normalized = 1 + ((vec - vec_min) * 254 / (vec_max - vec_min))
    return np.round(normalized).astype(int)

def to_binary_array(num):
    """Convert integer to 8-bit binary array"""
    # Convert to binary string and pad to 8 bits
    binary = format(int(num), f'0{BITS_PER_NUMBER}b')
    # Convert to array of 0s and 1s
    return np.array([int(bit) for bit in binary])

def verify_binary_range(binary_arr):
    """Verify that binary values are only 0 or 1"""
    unique_values = np.unique(binary_arr)
    print(f"Unique values in binary array: {unique_values}")
    assert len(unique_values) <= 2 and all(v in [0, 1] for v in unique_values), "Binary array contains non-binary values!"

def verify_numeric_range(numeric_arr):
    """Verify that numeric values are within 8-bit range"""
    min_val, max_val = numeric_arr.min(), numeric_arr.max()
    print(f"Numeric range: [{min_val}, {max_val}]")
    assert 1 <= min_val and max_val <= 255, "Numeric values outside 8-bit range!"

def convert_to_binary(embeddings):
    """Convert embedding array to binary representation"""
    # For each number in embeddings, convert to 8-bit binary
    # This will expand each 100-dim vector to 800-dim binary vector
    binary_data = []
    for vec in embeddings:
        binary_vec = np.concatenate([to_binary_array(num) for num in vec])
        binary_data.append(binary_vec)
    return np.array(binary_data)

def get_bert_embedding(text, tokenizer, model):
    """Get BERT embedding for a text"""
    # Tokenize and prepare input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Get BERT embedding
    with torch.no_grad():
        outputs = model(**inputs)
        # Use [CLS] token embedding as sentence representation
        embeddings = outputs.last_hidden_state[:, 0, :]
        
    # Convert to numpy and reduce dimensions if needed
    embedding = embeddings.numpy()[0]
    if len(embedding) > EMBEDDING_DIM:
        # Use PCA or simply take first EMBEDDING_DIM components
        embedding = embedding[:EMBEDDING_DIM]
    
    return embedding

def embed_sentence(text, tokenizer, model):
    """Convert a sentence to integer vector using BERT"""
    # Get BERT embedding
    vec = get_bert_embedding(preprocess(text), tokenizer, model)
    # Convert to positive integers
    return normalize_vector(vec)

def main():
    print('Loading BERT model and tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    print('Loading dataset...')
    df = pd.read_csv(DATASET_PATH)
    sentences = df['Sentence'].astype(str).tolist()
    labels = df['Label'].tolist()

    print('Building vocabulary...')
    vocab = build_vocab(sentences)
    print(f'Total vocabulary size: {len(vocab)}')
    save_vocab(vocab, VOCAB_PATH)
    
    print('\nEmbedding sentences...')
    embedded = []
    for sent in tqdm(sentences, desc='Embedding'):
        vec = embed_sentence(sent, tokenizer, model)
        embedded.append(vec)

    print('\nSaving embeddings...')
    # Save regular embeddings
    embedded_arr = np.vstack(embedded)
    print(f'Regular embedding shape: {embedded_arr.shape}')
    verify_numeric_range(embedded_arr)
    
    out_df = pd.DataFrame(embedded_arr)
    out_df['Label'] = labels
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f'Saved numeric embeddings to {OUTPUT_PATH}')

    # Convert to binary and save
    print('\nConverting to binary representation...')
    binary_arr = convert_to_binary(embedded_arr)
    print(f'Binary embedding shape: {binary_arr.shape}')
    verify_binary_range(binary_arr)
    
    binary_df = pd.DataFrame(binary_arr)
    binary_df['Label'] = labels
    binary_df.to_csv(BINARY_OUTPUT_PATH, index=False)
    print(f'Saved binary embeddings to {BINARY_OUTPUT_PATH}')

    # Print summary
    print('\nSummary:')
    print(f'- Original embeddings: {embedded_arr.shape[0]} samples x {embedded_arr.shape[1]} dimensions')
    print(f'- Binary embeddings: {binary_arr.shape[0]} samples x {binary_arr.shape[1]} dimensions')
    print(f'- Each number converted to {BITS_PER_NUMBER} bits')
    print(f'- Value range: 1 to 255 (fits in {BITS_PER_NUMBER} bits)')

if __name__ == '__main__':
    main()
