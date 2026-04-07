from sklearn.metrics.pairwise import cosine_similarity
from helper.models import Models
import numpy as np
import re
import os


class PromptPreprocessor:

    def __init__(self):
        
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        
        self.model = Models.get_encoder()

        self.SIGN_MAP = {
            "high": 1,
            "latest": 1,
            "low": -1,
            "old": -1,
            "short": 1,
            "medium": 1,
            "long": 1
        }

        self.filter_phrases = {

        "engagement": {

            "high": [
                "most engaging",
                "high engagement",
                "people interact with a lot",
                "with strong engagement",
                "interactive",
                "people enjoy watching",
                "that keep viewers engaged",
                "highly engaging content"
            ],

            "low": [
                "low engagement",
                "less engaging",
                "little interaction",
                "people do not engage much with",
                "quiet with few interactions",
                "low audience engagement"
                ]
            },


        "like_ratio": {

            "high": [
                "most liked",
                "highest like ratio",
                "videos with lots of likes",
                "highly liked",
                "videos people liked the most",
                "best liked content",
                "well liked",
                "videos with strong like ratio"
            ],

            "low": [
                "least liked",
                "low like ratio",
                "videos with few likes",
                "poorly liked",
                "videos with weak approval",
                "videos with low like percentage"
                ]
            },


        "comments": {

            "high": [
                "most commented",
                "videos with many comments",
                "videos people discuss a lot",
                "discussion heavy",
                "videos with active discussions",
                "videos with lots of feedback",
                "videos with strong audience discussion"
            ],

            "low": [
                "least commented",
                "few comments",
                "low discussion",
                "videos with little discussion",
                "videos people do not comment much on",
                "videos with minimal feedback"
                ]
            },


        "recency": {

            "latest": [
                "latest",
                "new",
                "recent uploads",
                "newest",
                "fresh content",
                "recently uploaded",
                "latest content",
                "newly released"
            ],

            "old": [
                "old",
                "classic",
                "older",
                "earlier",
                "past uploads",
                "archived",
                "old content",
                "classic tutorials",
                "legacy"
                ]
            },


        "popularity": {

            "high": [
                "popular",
                "trending",
                "viral",
                "most popular",
                "famous",
                "widely watched",
                "everyone is watching",
                "top trending",
                "high popularity"
                ],

            "low": [
                "least popular",
                "underrated",
                "hidden gem",
                "less popular",
                "not very popular",
                "low popularity",
                "unknown",
                "niche",
                "rarely watched"
                ]
            },

            "duration": {

                "short": [
                    "short",
                    "short video",
                    "quick video",
                    "quick tutorial",
                    "brief explanation",
                    "bite sized",
                    "bite sized tutorial",
                    "mini tutorial",
                    "short lesson",
                    "fast explanation",
                    "quick overview",
                    "in a few minutes",
                    "under 5 minutes",
                    "under 10 minutes",
                    "concise video",
                    "quick walkthrough",
                    "short and simple",
                    "compact tutorial"
                ],

                "medium": [
                    "medium length",
                    "standard length",
                    "regular tutorial",
                    "normal duration",
                    "moderate length",
                    "full tutorial",
                    "complete explanation",
                    "typical tutorial",
                    "detailed but not too long",
                    "balanced duration",
                    "around 10 to 20 minutes",
                    "mid length video",
                    "moderate tutorial",
                    "standard video length"
                ],

                "long": [
                    "long",
                    "long video",
                    "long tutorial",
                    "deep dive",
                    "in depth tutorial",
                    "full length lecture",
                    "extended tutorial",
                    "comprehensive guide",
                    "detailed walkthrough",
                    "full course",
                    "lecture style video",
                    "over 30 minutes",
                    "hour long video",
                    "long detailed explanation",
                    "deep technical explanation",
                    "masterclass",
                    "complete course"
                ]
            }

        }

        self.filter_vectors = self.build_filter_vectors()

    def clean_text(self, text: str):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def embed(self, text: str):
        return self.model.encode([text])[0]

    def build_filter_vectors(self):
        vectors = {}
        for category, values in self.filter_phrases.items():
            vectors[category] = {}
            for label, phrases in values.items():
                emb = self.model.encode(phrases)
                vectors[category][label] = np.mean(emb, axis=0)
        return vectors

    def extract_filters(self, prompt_vec, threshold=0.60):

        best_filter = {}
        
        for category, labels in self.filter_vectors.items():
        
            best_label = ""
            best_score = 0
        
            for label, vec in labels.items():
    
                sim = cosine_similarity([prompt_vec], [vec])[0][0]
                
                if sim > best_score:
                    best_score = sim
                    best_label = label

            best_filter[category] = {best_label : best_score if best_score >= threshold else 0}  #{Each category : {best label: best score}}

        return best_filter

    def compute_weights(self, filters: dict):
        
        weights = dict()

        for filter_name, result in filters.items():

            signal, score = list(result.items())[0]
            
            sign = self.SIGN_MAP[signal]
            
            weights[filter_name] = score * sign

        if all(w == 0 for w in weights.values()):

            return None

        weights = self.normalize_weights(weights)

        return weights
    
    def normalize_weights(self, weights: dict):

        total = sum(abs(weight) for weight in weights.values())
        
        if total == 0:

            return weights
        
        normalized = {
            filter_name: weight/total for filter_name, weight in weights.items()
        }

        return normalized

    def preprocess(self, user_prompt: str):
        
        cleaned = self.clean_text(user_prompt)
        
        prompt_vec = self.embed(cleaned)

        best = self.extract_filters(prompt_vec)

        duration = best.pop("duration", {})

        weights = self.compute_weights(best)

        return ({
            "raw_prompt": user_prompt,
            "cleaned_prompt": cleaned,
            "best_filters": best,
            "weights": weights
        }, prompt_vec, duration)

if __name__ == "__main__":
    
    pp = PromptPreprocessor()
    
    meta, vec, duration = pp.preprocess("Short form Machine Learning tutorials.")
    
    print("\nRaw Prompt:")
    print(meta["raw_prompt"])

    print("\nCleaned Prompt:")
    print(meta["cleaned_prompt"])
    
    print("\nFilter to be applied:")
    print(meta["best_filters"])

    print("\nWeights:")
    print(meta["weights"])

    print("\nDuration:")
    print(duration)