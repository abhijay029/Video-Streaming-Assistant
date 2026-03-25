from sentence_transformers import SentenceTransformer, CrossEncoder

class Models:

    @staticmethod
    def get_encoder():

        encoder_name = "BAAI/bge-large-en-v1.5"
        
        return SentenceTransformer(model_name_or_path = encoder_name)

    @staticmethod
    def get_reranker():
        
        reranker_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        return CrossEncoder(reranker_name)