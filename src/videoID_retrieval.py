import faiss
import numpy as np
import os
import pandas as pd
from helper.models import Models
from pprint import pprint
from helper.dataset import Dataset

class VideoIDRetriever:
    
    def __init__(self, vecDB_path: str, dataframe: pd.DataFrame):

        if not os.path.exists(vecDB_path):
            raise FileNotFoundError(f"The following index path does not exist: \n{vecDB_path}")

        self.index = faiss.read_index(vecDB_path)

        if self.index.ntotal == 0:
            raise ValueError(f"The following index does not contain any vectors: \n{vecDB_path}")
        
        self.df = dataframe
        
        self.videoIDs = self.df['video_id'].to_list()

        self.texts = (
            "Title: " + self.df["title"].fillna("") + "\n" +
            "Description: " + self.df["description"].fillna("") + "\n" +
            "Tags: " + self.df["tags"].fillna("")
            ).tolist()
        
        if len(self.videoIDs) != self.index.ntotal:
            raise ValueError(f"""Mismatch between number of video ids and number of embeddings:
                             \nvideo IDs: {len(self.videoIDs)}
                             \nindex: {self.index.ntotal}""")

        self.reranker = Models.get_reranker()


    def get_videoIDs(self, prompt: str,  prompt_vec: np.ndarray , k: int = 10):
        
        prompt_vec = prompt_vec.astype("float32")

        prompt_vec = prompt_vec.reshape(1, -1)

        faiss.normalize_L2(prompt_vec)

        distances, indices = self.index.search(prompt_vec, min(k, self.index.ntotal))

        selected_idx = indices[0]

        selected_ids = [self.videoIDs[index] for index in selected_idx]

        selected_text = [self.texts[index] for index in selected_idx]

        pairs = [(prompt, doc) for doc in selected_text]

        cross_scores = self.reranker.predict(pairs)

        faiss_scores = distances.reshape(-1)

        scores = [{"cross": cross_scores[i], "faiss": faiss_scores[i]} for i in range(len(selected_ids))]

        results = dict(zip(selected_ids, scores))

        results = dict(sorted(results.items(), key = lambda item: item[1]["cross"], reverse = True))
        
        return results


if __name__ == "__main__":
    
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    VEC_DATABASE_PATH = "Dataset\\video_index.faiss"
    df = Dataset.get_dataframe()

    model = Models.get_encoder()
    
    prompt = "I want videos of pets and pet care."
    
    prompt_vec = model.encode(prompt)

    ret = VideoIDRetriever(vecDB_path = VEC_DATABASE_PATH, dataframe = df)
    
    results = ret.get_videoIDs(prompt, prompt_vec = prompt_vec, k = 3)

    print(results)
