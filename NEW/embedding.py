import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

df = pd.read_csv("../En-Ba-Dataset(20k_4)/dataset.csv")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return tokens

embeddings_index = {}
with open("glove.6B.50d.txt", encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = vector

def text_to_vector(tokens):
    vectors = [embeddings_index[w] for w in tokens if w in embeddings_index]
    if len(vectors) == 0:
        return np.zeros(50)
    return np.mean(vectors, axis=0)

df["tokens"] = df["text"].apply(preprocess)
df["vector"] = df["tokens"].apply(text_to_vector)

vector_df = pd.DataFrame(df["vector"].tolist(), index=df.index)
vector_df.columns = [f"feature_{i}" for i in range(vector_df.shape[1])]

final_df = pd.concat([vector_df, df["label"]], axis=1)
final_df.to_csv("dataset_numeric.csv", index=False)
