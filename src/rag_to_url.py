import pandas as pd
from helper.dataset import Dataset

class RAGFetcher:
    def __init__(self, dataframe: pd.DataFrame, cross_scores: list = None, videoIDs: list = ["tkQwDzaarlM", "m-xcxqjjOwB", "mgKTtF-GswP", "G25yIun-Isc", "7LB6MSxROy8"]):
        self.df = dataframe
        self.df = self.df[self.df["video_id"].isin(videoIDs)]
        self.cross_scores = cross_scores

    def _format_duration(self, seconds):
        if pd.isna(seconds):
            return ""
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def get_rag_results(self) -> dict:
        """Convert CSV to RAG format with video URLs"""
        video_ids = self.df['video_id'].tolist()
        metadatas = []

        for _, row in self.df.iterrows():
            metadata = {
                'video_id': row['video_id'],
                'title': row['title'],
                'description': row['description'],
                'channel': row['channel_title'],
                'views': row['view_count'],
                'likes': row['like_count'],
                'duration': self._format_duration(row['duration_seconds']),
                'upload_date': row['published_at'],
                'like_view_ratio': row['like_view_ratio'],
                'comments_count': row['comment_count'],
                'year': row['published_at'].year,
                'duration_category': row['duration_category'],
                'topic': row['category_id'],
                'url': f"https://www.youtube.com/watch?v={row['video_id']}",
                'embed_url': f"https://www.youtube.com/embed/{row['video_id']}"
                
            }
            metadatas.append(metadata)

        return {
            'ids': [video_ids],
            'distances': [self.cross_scores],
            'metadatas': [metadatas]
        }

    def save_to_json(self, output_path: str):
        
        """Save RAG results to JSON file"""
        
        import json

        results = self.get_rag_results()
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved to {output_path}")


if __name__ == "__main__":

    df = Dataset.get_dataframe()

    fetcher = RAGFetcher(dataframe = df)
    results = fetcher.get_rag_results()

    print("\nFirst 3 videos with URLs:")
    for i, video in enumerate(results['metadatas'][0][:3]):
        print(f"{i+1}. {video['title']}")
        print(f"   URL: {video['url']}\n")


