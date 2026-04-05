import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from helper.dataset import Dataset

class Trending:

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    """## Prepare Data for Scoring

    ### Subtask:
    Calculate a 'recency' feature from the `published_at` column. It should represent how recent a video is, with newer videos having higher recency. Also, ensure numerical columns used for scoring are clean and ready for normalization.

    **Reasoning**:
    I will calculate the 'recency' feature by first determining the maximum `published_at` date, then calculating `days_since_latest` for each video, and finally deriving the `recency` score. After this, I will identify and confirm the data types of the numerical columns designated for scoring.
    """

    def get_trending_videos(self):

        # 1. Find the maximum published_at date
        max_published_at = self.df['published_at'].max()

        # Calculate days_since_latest
        self.df['days_since_latest'] = (max_published_at - self.df['published_at']).dt.days

        # 2. Create the 'recency' score
        self.df['recency'] = self.df['days_since_latest'].max() - self.df['days_since_latest']

        # 3. Identify numerical columns for scoring and confirm their data types
        scoring_columns = ['view_count', 'like_count', 'comment_count', 'like_view_ratio', 'recency']

        """## Normalize Metrics

        ### Subtask:
        Normalize `view_count`, `like_count`, `comment_count`, `like_view_ratio`, and the 'recency' metric. Use a suitable normalization technique (e.g., Min-Max Scaling) to bring all values to a similar scale (e.g., 0 to 1) for fair comparison.

        **Reasoning**:
        I will import the `MinMaxScaler`, define the columns to normalize, apply Min-Max Scaling to them, and then display the head of the DataFrame to show the normalized values.
        """

        # Define the columns to normalize
        columns_to_normalize = ['view_count', 'like_count', 'comment_count', 'like_view_ratio', 'recency']

        # Initialize MinMaxScaler
        scaler = MinMaxScaler()

        # Apply Min-Max Scaling to the selected columns
        self.df[columns_to_normalize] = scaler.fit_transform(self.df[columns_to_normalize])

        """## Calculate Trending Score

        ### Subtask:
        Implement a weighted scoring formula to combine the normalized metrics: `view_count`, `like_count`, `comment_count`, `like_view_ratio`, and 'recency'. Prioritize engagement (likes and comments) followed by recency and then views, as specified. Add this as a new column `trending_score` to the DataFrame.

        **Reasoning**:
        I will define weights for the normalized metrics, calculate the `trending_score` using a weighted sum, add it as a new column to the DataFrame, and then display the head of the DataFrame to verify the results as per the subtask instructions.
        """

        # Define weights for each normalized metric based on prioritization
        # Prioritize engagement (likes and comments) followed by recency and then views
        weights = {
            'view_count': 0.1,
            'like_count': 0.3,
            'comment_count': 0.3,
            'like_view_ratio': 0.2,
            'recency': 0.1
        }

        # Calculate the trending_score
        self.df['trending_score'] = (
            self.df['view_count'] * weights['view_count'] +
            self.df['like_count'] * weights['like_count'] +
            self.df['comment_count'] * weights['comment_count'] +
            self.df['like_view_ratio'] * weights['like_view_ratio'] +
            self.df['recency'] * weights['recency']
        )

        """## Identify Top 20 Trending Videos

        ### Subtask:
        Sort the DataFrame by the newly created `trending_score` in descending order. Select the top 20 videos and display only the `title`, `channel_title`, `view_count`, `like_count`, `comment_count`, and `trending_score` columns.

        **Reasoning**:
        I will sort the DataFrame by `trending_score` in descending order, select the top 20 videos, and display the specified columns to identify the trending videos.
        """

        top_20_trending_videos = self.df.sort_values(by='trending_score', ascending=False).head(20)


        return top_20_trending_videos


if __name__ == "__main__":

    df =Dataset.get_dataframe()

    trend = Trending(dataframe = df)
    
    videos = trend.get_trending_videos()

    print(type(videos))

    print(videos[["video_id", "title", "description"]].head(20))