import faiss
import numpy as np
import pandas as pd
import os
from models import Models
from dataset import Dataset

def get_text_embeddings(dataframe: pd.DataFrame):

    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    model = Models.get_encoder()

    df = dataframe

    videoIDs = df["video_id"].tolist()

    texts = (
            "Title: " + df["title"].fillna("") + "\n" +
            "Description: " + df["description"].fillna("") + "\n" +
            "Tags: " + df["tags"].fillna("")
            ).tolist()

    embeddings = model.encode(texts).astype("float32")

    print(f"Embeddings created for {len(embeddings)} videos")

    return embeddings, videoIDs


def create_save_index(name: str, dataframe: pd.DataFrame):

    embeddings, videoIDs = get_text_embeddings(dataframe = dataframe)

    dimension = embeddings.shape[1]

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    if index.ntotal == 0:
        raise ValueError("Index contains 0 embeddings")

    if not name.endswith(".faiss"):
        name = name.strip() + ".faiss"

    faiss.write_index(index, name)

    print(f"Index created with name: {name}")
    print(f"Total vectors stored: {index.ntotal}")

    return index, videoIDs

if __name__ == '__main__':

    df = Dataset.get_dataframe()
    
    create_save_index(name = "video_index", dataframe = df)