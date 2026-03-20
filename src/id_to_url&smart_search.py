import pandas as pd

# Load the dummy dataset
df = pd.read_csv('/content/youtube_videos_dataset.csv')

"""Now that we have loaded and inspected the dataset, I will transform it into the `rag_results` format expected by the `your_main.py` script. The `rag_results` dictionary needs to have `ids` and `metadatas` keys, where `ids` is a list of video IDs and `metadatas` is a list of dictionaries containing video information."""

# Prepare the data in the rag_results format
# Assuming the CSV has a 'video_id' column for IDs and other columns for metadata

# We need to map the dataframe columns to the expected metadata keys
# Based on the structure in your_main.py, the metadata keys are:
# 'title', 'description', 'channel', 'views', 'likes', 'duration', 'upload_date'

video_ids = df['video_id'].tolist()

metadatas = []
for index, row in df.iterrows():
    # Function to format duration from seconds to HH:MM:SS or MM:SS
    def format_duration(seconds):
        if pd.isna(seconds):
            return ""
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
        else:
            return f"{minutes:02d}:{remaining_seconds:02d}"

    metadata = {
        'title': row['title'],
        'description': row['description'],
        'channel': row['channel_name'],
        'views': row['views'],
        'likes': row['likes'],
        'duration': format_duration(row['duration_seconds']), # Formatted duration
        'upload_date': row['upload_date'] # Using 'upload_date' from df
    }
    metadatas.append(metadata)

# Construct the rag_results dictionary
new_rag_results = {
    'ids': [video_ids],
    'distances': [[0.0] * len(video_ids)], # Dummy distances, as they are not in the CSV
    'metadatas': [metadatas]
}

# Display the first few elements of the new_rag_results for verification
print("Generated rag_results (first few):")
print(f"IDs: {new_rag_results['ids'][0][:3]}")
print(f"Metadatas: {new_rag_results['metadatas'][0][:2]}")

"""Now, I will modify the `your_main.py` script to use this `new_rag_results` instead of the hardcoded sample data. I will also make the `rag_results` a global variable or pass it to `main()` function, so that it can be accessed by the `main()` function."""

# your_main.py
# This is YOUR code that will use your friend's RAG output

from video_display import VideoDisplay, VideoPlayer

def main(rag_results_data):
    # Initialize your display module
    display = VideoDisplay()
    player = VideoPlayer()

    # 1. Get the RAG results from your friend's code
    # Your friend will call this function and give you rag_results
    # rag_results = friend_search_function(user_query)

    # Using the dynamically loaded rag_results
    rag_results = rag_results_data

    # 2. Format the results using your display module
    formatted_videos = display.format_results(
        rag_results,
        query="your search query"
    )

    # 3. Display to user
    display.display_console(formatted_videos)

    # 4. If user selects a video, you can play it
    if formatted_videos:
        selected = formatted_videos[0]  # User selects first video
        print(f"\n▶️ Now playing: {selected['title']}")
        print(f"Watch here: {selected['url']}")

        # You can also get embed URL for web playback
        embed_url = player.get_embed_url(selected['video_id'])
        print(f"Embed URL: {embed_url}")

if __name__ == "__main__":
    # Pass the global new_rag_results to the main function
    main(new_rag_results)

# video_ranker.py - Fixed for multiple video output
import numpy as np
from typing import List, Dict, Any

class VideoRanker:
    def __init__(self):
        """Initialize the video ranker with your specific dataset columns"""
        pass

    def apply_filters(self,
                     rag_results: Dict[str, Any],
                     filter_type: str = 'engagement',
                     n_results: int = 10,  # Default to 10 videos
                     min_views: int = None,
                     min_likes: int = None,
                     min_engagement: float = None,
                     difficulty: str = None,
                     topic: str = None,
                     duration: str = None) -> Dict[str, Any]:
        """
        Apply various filters to RAG results using your actual columns

        Args:
            rag_results: Original results from FAISS (can have many videos)
            filter_type: 'engagement', 'like_ratio', 'recency', 'comments', 'balanced'
            n_results: Number of results to return (user can specify any number)
            min_views: Minimum views threshold
            min_likes: Minimum likes threshold
            min_engagement: Minimum engagement score threshold
            difficulty: Filter by difficulty level
            topic: Filter by topic
            duration: Filter by duration category
        """

        # Extract data from RAG results - these can have many videos
        video_ids = rag_results['ids'][0]           # List of all video IDs
        distances = rag_results['distances'][0]      # List of all distances
        metadatas = rag_results['metadatas'][0]      # List of all metadata

        print(f"📊 Total videos received from RAG: {len(video_ids)}")

        # Create list of videos with all your columns
        videos = []
        for i, (video_id, metadata, distance) in enumerate(zip(video_ids, metadatas, distances)):

            # Apply basic filters
            if min_views and metadata.get('views', 0) < min_views:
                continue
            if min_likes and metadata.get('likes', 0) < min_likes:
                continue
            if min_engagement and metadata.get('engagement_score', 0) < min_engagement:
                continue
            if difficulty and metadata.get('difficulty_level') != difficulty:
                continue
            if topic and metadata.get('topic') != topic:
                continue
            if duration and metadata.get('duration_category') != duration:
                continue

            # Relevance score from semantic search
            relevance_score = 1 - distance

            # Use pre-calculated metrics from your dataset
            like_view_ratio = metadata.get('like_view_ratio', 0)
            engagement_score = metadata.get('engagement_score', 0)
            comments = metadata.get('comments_count', 0)
            views = metadata.get('views', 0)
            likes = metadata.get('likes', 0)

            # Calculate final score based on filter type
            if filter_type == 'engagement':
                # Use your pre-calculated engagement score
                final_score = engagement_score

            elif filter_type == 'like_ratio':
                # Use your pre-calculated like/view ratio
                final_score = like_view_ratio

            elif filter_type == 'comments':
                # Normalize comments count
                final_score = np.log1p(comments)

            elif filter_type == 'recency':
                # Use year as simple recency metric (newer = higher score)
                year = metadata.get('year', 2020)
                final_score = (year - 2020) / 5  # Normalize 2020-2025 to 0-1

            elif filter_type == 'popularity':
                # Combined popularity metric
                final_score = (0.4 * np.log1p(views) +
                             0.4 * np.log1p(likes) +
                             0.2 * np.log1p(comments))

            elif filter_type == 'balanced':
                # Combine relevance with engagement
                final_score = (0.3 * relevance_score +
                             0.7 * engagement_score)

            elif filter_type == 'topic_expert':
                # For educational content - combine difficulty with engagement
                difficulty_weight = {
                    'Beginner': 0.3,
                    'Intermediate': 0.5,
                    'Advanced': 0.7
                }.get(metadata.get('difficulty_level', 'Beginner'), 0.3)

                final_score = (0.5 * engagement_score +
                             0.3 * difficulty_weight +
                             0.2 * relevance_score)
            else:
                final_score = relevance_score

            videos.append({
                'video_id': video_id,
                'metadata': metadata,
                'distance': distance,
                'relevance_score': relevance_score,
                'engagement_score': engagement_score,
                'like_view_ratio': like_view_ratio,
                'comments': comments,
                'views': views,
                'likes': likes,
                'final_score': final_score
            })

        print(f"🎯 Videos after filters: {len(videos)}")

        # Sort by final score (highest first)
        videos.sort(key=lambda x: x['final_score'], reverse=True)

        # Take only the number of results user asked for
        if n_results and len(videos) > n_results:
            videos = videos[:n_results]
            print(f"✅ Showing top {n_results} videos")
        else:
            print(f"✅ Showing all {len(videos)} videos")

        # Reconstruct results in the same format as RAG output
        filtered_results = {
            'ids': [[v['video_id'] for v in videos]],
            'distances': [[v['distance'] for v in videos]],
            'metadatas': [[v['metadata'] for v in videos]],
            'scores': {
                'final_scores': [v['final_score'] for v in videos],
                'engagement_scores': [v['engagement_score'] for v in videos],
                'like_view_ratios': [v['like_view_ratio'] for v in videos],
                'relevance_scores': [v['relevance_score'] for v in videos]
            }
        }

        return filtered_results

# Enhanced Display Class for Your Dataset
class EnhancedVideoDisplay:
    def __init__(self):
        self.ranker = VideoRanker()

    def display_with_filters(self, rag_results, query=None, filter_type='engagement', n_results=10):
        """Display multiple videos with your specific columns"""

        filtered_results = self.ranker.apply_filters(
            rag_results,
            filter_type=filter_type,
            n_results=n_results
        )

        print("\n" + "="*100)
        print(f"🔍 Results for: '{query}'")
        print(f"📊 Filter: {filter_type.upper()} mode")
        print(f"📺 Showing {len(filtered_results['ids'][0])} videos")
        print("="*100 + "\n")

        if not filtered_results['ids'][0]:
            print("❌ No videos match the filter criteria")
            return

        # Display each video with its rank
        for i, (video_id, metadata, final_score) in enumerate(zip(
            filtered_results['ids'][0],
            filtered_results['metadatas'][0],
            filtered_results['scores']['final_scores']
        )):
            watch_url = f"https://www.youtube.com/watch?v={video_id}"

            # Extract your columns
            title = metadata.get('title', 'Untitled')
            channel = metadata.get('channel', 'Unknown')
            views = metadata.get('views', 0)
            likes = metadata.get('likes', 0)
            comments = metadata.get('comments_count', 0)
            like_ratio = metadata.get('like_view_ratio', 0) * 100  # Convert to percentage
            engagement = metadata.get('engagement_score', 0) * 100
            difficulty = metadata.get('difficulty_level', 'N/A')
            topic = metadata.get('topic', 'N/A')
            duration_cat = metadata.get('duration_category', 'N/A')
            year = metadata.get('year', 'N/A')
            tags = metadata.get('tags', [])

            # Medal emoji for top 3
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."

            print(f"{rank_emoji} 🎬 {title}")
            print(f"   🔗 {watch_url}")
            print(f"   👤 {channel}  |  📚 {topic}  |  📊 {difficulty}")
            print(f"   ⏱️ {duration_cat}  |  📅 {year}")
            print(f"   📈 Views: {self._format_number(views)}  |  👍 Likes: {self._format_number(likes)}  |  💬 Comments: {self._format_number(comments)}")

            # Show key metrics
            print(f"   ❤️ Like Ratio: {like_ratio:.1f}%  |  ⭐ Engagement: {engagement:.1f}%")

            # Show tags if available
            if tags and len(tags) > 0:
                tag_str = ", ".join(tags[:3])  # Show first 3 tags
                print(f"   🏷️ Tags: {tag_str}")

            print(f"   🎯 Quality Score: {final_score:.3f}")
            print("-"*100)

    def _format_number(self, num):
        """Format numbers with K/M suffix"""
        if num >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num/1_000:.1f}K"
        return str(num)

# Interactive Filter Interface
class VideoSearchInterface:
    def __init__(self):
        self.display = EnhancedVideoDisplay()

    def interactive_search(self, rag_results):
        """Let user choose filters interactively"""

        print("\n" + "="*60)
        print("🎯 VIDEO SEARCH FILTERS")
        print("="*60)

        # Show how many videos we have
        total_videos = len(rag_results['ids'][0])
        print(f"\n📊 Total videos available: {total_videos}")

        # Filter type selection
        print("\n📊 Choose filter type:")
        print("1. ⭐ Engagement Score (pre-calculated)")
        print("2. ❤️ Like/View Ratio (pre-calculated)")
        print("3. 💬 Most Comments")
        print("4. 🆕 Most Recent (by year)")
        print("5. 🔥 Popularity (views + likes + comments)")
        print("6. ⚖️ Balanced (relevance + engagement)")
        print("7. 🎓 Topic Expert (by difficulty + engagement)")

        filter_choice = input("\nEnter choice (1-7): ").strip()

        filter_map = {
            '1': 'engagement',
            '2': 'like_ratio',
            '3': 'comments',
            '4': 'recency',
            '5': 'popularity',
            '6': 'balanced',
            '7': 'topic_expert'
        }

        filter_type = filter_map.get(filter_choice, 'engagement')

        # Number of results
        n_results = input(f"\nHow many videos to show? (1-{total_videos}, default 10): ").strip()
        n_results = int(n_results) if n_results.isdigit() and 1 <= int(n_results) <= total_videos else 10

        # Optional filters
        print("\n🔧 Optional filters (press Enter to skip):")

        # Difficulty filter
        print("\nDifficulty levels:")
        print("1. Beginner")
        print("2. Intermediate")
        print("3. Advanced")
        diff_choice = input("Choose difficulty (1-3): ").strip()
        diff_map = {'1': 'Beginner', '2': 'Intermediate', '3': 'Advanced'}
        difficulty = diff_map.get(diff_choice)

        # Duration filter
        print("\nDuration categories:")
        print("1. Short")
        print("2. Medium")
        print("3. Long")
        dur_choice = input("Choose duration (1-3): ").strip()
        dur_map = {'1': 'Short', '2': 'Medium', '3': 'Long'}
        duration = dur_map.get(dur_choice)

        # Minimum thresholds
        min_views = input("\nMinimum views (e.g., 1000): ").strip()
        min_views = int(min_views) if min_views.isdigit() else None

        min_likes = input("Minimum likes (e.g., 100): ").strip()
        min_likes = int(min_likes) if min_likes.isdigit() else None

        min_engagement = input("Minimum engagement % (e.g., 5): ").strip()
        min_engagement = float(min_engagement)/100 if min_engagement else None

        # Apply all filters
        filtered_results = self.display.ranker.apply_filters(
            rag_results,
            filter_type=filter_type,
            n_results=n_results,
            min_views=min_views,
            min_likes=min_likes,
            min_engagement=min_engagement,
            difficulty=difficulty,
            duration=duration
        )

        # Display results
        query = input("\n🔍 Enter search query (optional): ").strip()
        self.display.display_with_filters(
            filtered_results,
            query=query or "Custom Search",
            filter_type=filter_type,
            n_results=n_results
        )

# Example usage with multiple videos
def main():
    # This simulates your friend's RAG output with MULTIPLE videos
    # In reality, your friend's code will return however many videos ChromaDB finds

    # Create sample data with 15 videos (using your actual data structure)
    sample_videos = []
    for i in range(15):
        sample_videos.append({
            'title': f'Video Title {i+1}',
            'channel': f'Channel {i+1}',
            'views': 100000 * (i+1),
            'likes': 5000 * (i+1),
            'comments_count': 100 * (i+1),
            'duration_category': ['Short', 'Medium', 'Long'][i % 3],
            'year': 2020 + (i % 5),
            'topic': ['Gaming', 'Cooking', 'Tech', 'Travel', 'Fitness'][i % 5],
            'difficulty_level': ['Beginner', 'Intermediate', 'Advanced'][i % 3],
            'like_view_ratio': 0.01 + (i * 0.005),
            'engagement_score': 0.02 + (i * 0.003),
            'tags': ['tag1', 'tag2', 'tag3']
        })

    # Create RAG results with all 15 videos
    rag_results = {
        'ids': [[f'vid{i+1}' for i in range(15)]],
        'distances': [[0.1 + (i*0.02) for i in range(15)]],
        'metadatas': [sample_videos]
    }

    print("="*60)
    print("🎬 VIDEO RECOMMENDATION SYSTEM")
    print("="*60)
    print(f"📊 Total videos in database: {len(rag_results['ids'][0])}")

    # Simple display - show top 8 videos
    display = EnhancedVideoDisplay()
    display.display_with_filters(
        rag_results,
        query="popular videos",
        filter_type='popularity',
        n_results=8  # User asked for 8 videos
    )

    # Or use interactive mode
    # interface = VideoSearchInterface()
    # interface.interactive_search(rag_results)

if __name__ == "__main__":
    main()

