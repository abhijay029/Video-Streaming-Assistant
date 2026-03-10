import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os


def get_title_desc_embeddings(dataset_path: str):

    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    model = SentenceTransformer("all-MiniLM-L6-v2")

    df = pd.read_csv(dataset_path)

    videoIDs = df["video_id"].tolist()

    texts = (df["title"] + " " + df["description"]).tolist()

    embeddings = model.encode(texts).astype("float32")

    print(f"Embeddings created for {len(embeddings)} videos")

    return embeddings, videoIDs


def create_save_index(name: str, dataset_path: str):

    embeddings, videoIDs = get_title_desc_embeddings(dataset_path)

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
    DATASET_PATH = "Dataset/youtube_videos_dataset.csv"

    create_save_index(name = "video_index", dataset_path = DATASET_PATH)