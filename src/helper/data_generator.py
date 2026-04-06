import pandas as pd
import isodate
from googleapiclient.discovery import build
from tqdm import tqdm
from dotenv import load_dotenv
import os
import emoji
import re
import numpy as np

load_dotenv(".env")

API_KEY = os.getenv("YOUTUBE_API_KEY")

youtube = build(
    "youtube",
    "v3",
    developerKey=API_KEY
)

def preprocess_text(text):

    # Convert emojis
    text = emoji.demojize(text)

    # Remove emoji markers
    text = text.replace(":", " ")

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Convert hashtags
    text = text.replace("#", " ")

    # Remove extra punctuation
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# Search videos by keyword
def search_videos(query, max_results=200):

    video_ids = []
    next_page_token = None

    while len(video_ids) < max_results:

        request = youtube.search().list(
            part="id",
            q=query,
            type="video",
            maxResults=50,
            pageToken=next_page_token
        )

        response = request.execute()

        for item in response["items"]:
            video_ids.append(
                item["id"]["videoId"]
            )

        next_page_token = response.get(
            "nextPageToken"
        )

        if not next_page_token:
            break

    return video_ids[:max_results]


# Get video metadata
def get_video_details(video_ids):

    data = []

    for i in tqdm(range(0, len(video_ids), 50)):

        request = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(video_ids[i:i+50])
        )

        response = request.execute()

        for item in response["items"]:

            snippet = item["snippet"]
            stats = item["statistics"]
            content = item["contentDetails"]

            duration_iso = content["duration"]

            # Convert ISO duration
            duration_seconds = int(
                isodate.parse_duration(
                    duration_iso
                ).total_seconds()
            )

            # Duration category
            if duration_seconds < 240:
                duration_category = "short"
            elif duration_seconds < 1200:
                duration_category = "medium"
            else:
                duration_category = "long"

            view_count = int(stats.get("viewCount", 0))
            like_count = int(stats.get("likeCount", 0))

            like_view_ratio = (
                like_count / view_count
                if view_count > 0 else 0
            )

            popularity_score = (0.4 * np.log1p(view_count) + 0.4 * np.log1p(like_count) + 0.2 * np.log1p(int(stats.get("commentCount", 0))))

            title = preprocess_text(snippet.get("title", ""))
            description = preprocess_text(snippet.get("description", ""))
            channel_title = preprocess_text(snippet.get("channelTitle", ""))
            tags = snippet.get("tags", [])
            
            if tags:
            
                tags = [preprocess_text(tag) for tag in tags]
            
            else:

                tags = []

            combined_text = (
                title + " " +
                description + " " +
                " ".join(tags)
            )

            data.append({
                "video_id": item["id"],
                "title": title,
                "description": description,
                "channel_title": channel_title,
                "channel_id": snippet["channelId"],
                "published_at": snippet["publishedAt"],
                "view_count": view_count,
                "like_count": like_count,
                "comment_count": int(
                    stats.get("commentCount", 0)
                ),
                "duration_iso": duration_iso,
                "tags": tags,
                "category_id": snippet["categoryId"],
                "duration_seconds": duration_seconds,
                "duration_category": duration_category,
                "year": snippet["publishedAt"][:4],
                "like_view_ratio": like_view_ratio,
                "popularity_score": popularity_score,
                "combined_text": combined_text
            })

    return pd.DataFrame(data)


# Example queries
queries = [
    "machine learning",
    "python tutorial",
    "travel vlog"
]

all_ids = list()

for q in queries:
    all_ids.extend(
        search_videos(q, max_results=200)
    )

all_ids = list(set(all_ids))

df = get_video_details(all_ids)

df.to_csv(
    "real_youtube_dataset.csv",
    index=False
)

print("Dataset created successfully!")