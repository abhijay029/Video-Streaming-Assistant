import numpy as np
import pandas as pd
import os
import math
from tqdm import tqdm
from helper.dataset import Dataset
from helper.models import Models
from videoID_retrieval import VideoIDRetriever
from sklearn.metrics.pairwise import cosine_similarity
from main import RankedVideos

VEC_DATABASE_PATH = "Dataset/video_index.faiss"

K = 5   
NUM_QUERIES = 1

video_relevance_scores = dict()

def precision_at_k(retrieved, relevant, k):

    retrieved_k = retrieved[:k]
    
    relevant_set = set(relevant)

    hits = sum(1 for vid in retrieved_k if vid in relevant_set)

    return hits / k


def recall_at_k(retrieved, relevant, k):
    
    retrieved_k = retrieved[:k]

    relevant_set = set(relevant)

    hits = sum(1 for vid in retrieved_k if vid in relevant_set)

    if len(relevant_set) == 0:
        return 0

    return hits / len(relevant_set)


def reciprocal_rank(retrieved, relevant):

    relevant_set = set(relevant)

    for i, vid in enumerate(retrieved):
        if vid in relevant_set:
            return 1 / (i + 1)

    return 0


def dcg_at_k(relevance_scores, k):

    return sum( rel / math.log2(idx + 2) for idx, rel in enumerate(relevance_scores[:k]))


def ndcg_at_k(retrieved, query_str: str, k):

    relevance_scores = [video_relevance_scores[query_str][vid] for vid in retrieved]

    dcg = dcg_at_k(relevance_scores, k)

    ideal_scores = sorted(relevance_scores, reverse=True)

    idcg = dcg_at_k(ideal_scores, k)

    if idcg == 0:
        return 0

    return dcg / idcg


def generate_queries(df, num_queries=100):

    titles = df["title"].dropna().tolist()

    queries = titles[ : num_queries: 100]

    return queries


def generate_ground_truth(df, queries, encoder):

    ground_truth = {}

    texts = df["combined_text"].tolist()
    video_ids = df["video_id"].tolist()

    embedding_path = "video_embeddings.npy"

    if os.path.exists(embedding_path):

        print("Loading cached embeddings...")

        text_vector = np.load(embedding_path)

    else:

        print("Encoding video texts...")

        text_vector = encoder.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        np.save(embedding_path, text_vector)

    for query in tqdm(queries):

        query_vector = encoder.encode([query], normalize_embeddings=True)[0]

        scores = cosine_similarity([query_vector], text_vector)[0]

        id_relevance = list(zip(video_ids, scores))

        video_relevance_scores[query] = dict(id_relevance)

        top_n = 20

        sorted_pairs = sorted(id_relevance, key=lambda x: x[1], reverse=True)

        relevant_ids = [vid for vid, _ in sorted_pairs[:top_n]]

        ground_truth[query] = relevant_ids

    return ground_truth


def evaluate():

    print("Loading dataset...")

    df = Dataset.get_dataframe()

    print("Loading encoder model...")

    encoder = Models.get_encoder()

    print("Loading retriever...")

    retriever = VideoIDRetriever(vecDB_path=VEC_DATABASE_PATH, dataframe=df)

    print("Generating test queries...")

    queries = generate_queries(df, num_queries=NUM_QUERIES)

    print("Generating ground truth...")

    ground_truth = generate_ground_truth(df, queries, encoder = encoder)

    precision_scores = []
    recall_scores = []
    mrr_scores = []
    ndcg_scores = []

    print("Running evaluation...")

    for query in tqdm(queries):

        prompt_vec = encoder.encode([query], normalize_embeddings = True)[0]

        results = retriever.get_videoIDs(prompt=query, prompt_vec=prompt_vec, k=K)

        retrieved_ids = list(results.keys())

        relevant_ids = ground_truth[query]

        precision_scores.append(

            precision_at_k(retrieved_ids, relevant_ids, K)
        
        )

        recall_scores.append(

            recall_at_k(retrieved_ids, relevant_ids, K)
        
        )

        mrr_scores.append(
            reciprocal_rank(retrieved_ids, relevant_ids)
        )

        ndcg_scores.append(

            ndcg_at_k(retrieved_ids, query, K)
        
        )

    print("\n========== RESULTS ==========")

    print(
        f"Precision@{K}:",
        np.mean(precision_scores)
    )

    print(
        f"Recall@{K}:",
        np.mean(recall_scores)
    )

    print(
        "MRR:",
        np.mean(mrr_scores)
    )

    print(
        f"nDCG@{K}:",
        np.mean(ndcg_scores)
    )

if __name__ == "__main__":

    os.environ[
        "HF_HUB_DISABLE_SYMLINKS_WARNING"
    ] = "1"

    evaluate()