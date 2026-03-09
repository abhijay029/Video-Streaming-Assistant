from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

tf = SentenceTransformer("all-MiniLM-L6-v2")

phrase1 = "summarize this video"
phrase2 = "brief explanation"

vector1 = tf.encode(phrase1)
vector2 = tf.encode(phrase2)

sim = cosine_similarity([vector1], [vector2])
print(sim)