from sklearn.metrics.pairwise import cosine_similarity
from helper.models import Models
import numpy as np
import re
import os


class PromptPreprocessor:

    def __init__(self):
        
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        
        self.model = Models.get_encoder()

        self.intent_phrases = {
            "summary": [
                "summarize this video",
                "give me a summary",
                "brief explanation",
                "short overview"
            ],
            "qa": [
                "answer my question",
                "explain this",
                "what does this mean",
                "why does this happen"
            ],
            "recommendation": [
                "suggest videos",
                "recommend content",
                "what should I watch",
                "similar videos"
            ],
            "ranking": [
                "rank these videos",
                "best videos",
                "top content",
                "most useful videos"
            ],
            "channel": [
                "recommend channels",
                "best creators",
                "similar channels"
            ]
        }

        self.filter_phrases = {
            "popularity": {
                "high": ["popular", "viral", "trending", "famous"],
                "medium": ["moderately popular"],
                "low": ["less known", "underrated"]
            },
            "views": {
                "high": ["high views", "most viewed", "millions of views"],
                "medium": ["decent views"],
                "low": ["few views", "low views"]
            },
            "likes": {
                "high": ["high likes", "most liked"],
                "medium": ["moderate likes"],
                "low": ["low likes"]
            },
            "subscribers": {
                "high": ["big channel", "famous creator", "large audience"],
                "medium": ["growing channel"],
                "low": ["small creator", "new channel"]
            },
            "recency": {
                "latest": ["latest videos", "new videos", "recent uploads"],
                "old": ["old videos", "classic videos"]
            },
            "upload_time": {
                "morning": ["morning upload"],
                "afternoon": ["afternoon upload"],
                "evening": ["evening upload"],
                "night": ["night upload"]
            }
        }

        self.intent_vectors = self._build_intent_vectors()
        self.filter_vectors = self._build_filter_vectors()


    def _clean_text(self, text: str):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _embed(self, text: str):
        return self.model.encode([text])[0]


    def _build_intent_vectors(self):
        vectors = {}
        for intent, phrases in self.intent_phrases.items():
            emb = self.model.encode(phrases)
            vectors[intent] = np.mean(emb, axis=0) 
        return vectors

    def _build_filter_vectors(self):
        vectors = {}
        for category, values in self.filter_phrases.items():
            vectors[category] = {}
            for label, phrases in values.items():
                emb = self.model.encode(phrases)
                vectors[category][label] = np.mean(emb, axis=0)
        return vectors


    def detect_intent(self, prompt_vec):
        scores = {}
        for intent, vec in self.intent_vectors.items():
            sim = cosine_similarity([prompt_vec], [vec])[0][0]
            scores[intent] = sim

        best_intent = max(scores, key= lambda x: scores[x])
        return best_intent, scores

    def extract_filters(self, prompt_vec, threshold=0.55):

        extracted = {}
        best_filter = {}
        valid_filter = {}
        
        for category, labels in self.filter_vectors.items():
        
            best_label = None
            best_score = 0
            current_label = None
            current_score = 0
        
            for label, vec in labels.items():
            
                current_label = label
                sim = cosine_similarity([prompt_vec], [vec])[0][0]
                current_score = sim
                
                if sim > best_score:
                    best_score = sim
                    best_label = label
                
                extracted[category] = {label : sim} #{Each Category: {each label: score}}
                
                if sim > threshold:
                    valid_filter[category] = {label : sim} #{Each category: {valid label: score}}
            
            best_filter[category] = {best_label : best_score} #{Each category : {best label: best score}}

        return (extracted, valid_filter, best_filter)


    def preprocess(self, user_prompt: str):
        cleaned = self._clean_text(user_prompt)
        prompt_vec = self._embed(cleaned)

        intent, intent_scores = self.detect_intent(prompt_vec)
        extracted, valid, best = self.extract_filters(prompt_vec)

        return ({
            "raw_prompt": user_prompt,
            "cleaned_prompt": cleaned,
            "intent": intent,
            "intent_scores": intent_scores,
            "extracted_filters": extracted,
            "valid_filters":valid,
            "best_filters": best
            # "embedding": prompt_vec.tolist()
        }, prompt_vec)

if __name__ == "__main__":
    pp = PromptPreprocessor()
    meta, vec = pp.preprocess("Suggest me some CNN tutorials that are long form and rank them according to popularity")
    print(meta)