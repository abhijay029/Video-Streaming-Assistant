import faiss
import numpy as np
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

class VideoIDRetriever:
    
    def __init__(self, vecDB_path: str, dataset_path: str):

        if not os.path.exists(vecDB_path):
            raise FileNotFoundError(f"The following index path does not exist: \n{vecDB_path}")

        self.index = faiss.read_index(vecDB_path)

        if self.index.ntotal == 0:
            raise ValueError(f"The following index does not contain any vectors: \n{vecDB_path}")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"The following dataset path does not exist: \n{dataset_path}")
        
        self.videoIDs = pd.read_csv(dataset_path)['video_id'].to_list()

        if len(self.videoIDs) != self.index.ntotal:
            raise ValueError(f"""Mismatch between number of video ids and number of embeddings:
                             \nvideo IDs: {len(self.videoIDs)}
                             \nindex: {self.index.ntotal}""")
    
    def get_videoIDs(self, prompt_vec: np.ndarray , k: int = 10):
        
        prompt_vec = prompt_vec.astype("float32")

        prompt_vec = prompt_vec.reshape(1, -1)

        faiss.normalize_L2(prompt_vec)

        distances, indices = self.index.search(prompt_vec, min(k, self.index.ntotal))

        return ([self.videoIDs[i] for i in indices[0]], distances.reshape(-1,))


if __name__ == "__main__":
    
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    VEC_DATABASE_PATH = "Dataset\\video_index.faiss"
    DATABASE_PATH = "Dataset\\youtube_videos_dataset.csv"

    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    prompt = "I want to watch some tutorial for tensorflow CNN."
    
    prompt_vec = model.encode(prompt)

    ret = VideoIDRetriever(vecDB_path = VEC_DATABASE_PATH, dataset_path = DATABASE_PATH)
    
    videoIDs, distances = ret.get_videoIDs(prompt_vec = prompt_vec, k = 3)

    print(videoIDs)
    print(distances)
