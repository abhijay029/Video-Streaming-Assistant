from prompt_preprocessor import PromptPreprocessor
from videoID_retrieval import VideoIDRetriever
from pprint import pprint
from frame_extractor import FrameExtractor
from video_frame_intrpreter import FrameInterpreter
from video_query_responder import VideoAssistant
from rag_to_url import RAGFetcher
from searching_ranking import VideoRanker

class RankedVideos:
    
    def __init__(self):
        
        self.VEC_DB_PATH = "Dataset\\video_index.faiss"
        self.DATASET_PATH = "Dataset\\youtube_videos_dataset.csv"
    
        self.preprocessor = PromptPreprocessor()
        self.retriever = VideoIDRetriever(vecDB_path = self.VEC_DB_PATH, dataset_path = self.DATASET_PATH)
        
        self.ranker = VideoRanker()

    def get_ranked_videos(self, prompt: str):

        meta, vec = self.preprocessor.preprocess(prompt)

        print("meta information of the prompt:")
        pprint(meta)

        print("")
        
        print("Prompt vector: ", vec.shape)

        videoIDs, distances = self.retriever.get_videoIDs(vec, k = 5)

        print("")

        print("Video IDs Retrieved: ", len(videoIDs))
        print("Distances of title and description of videos from the user prompt: ", len(distances))
        
        #get the url
        self.url_fetcher = RAGFetcher(csv_path = self.DATASET_PATH, distances = distances, videoIDs = videoIDs)
        result = self.url_fetcher.get_rag_results()
        
        print("")

        print("Video IDs:", len(result["ids"][0]))
        print("Distances: ", len(result["distances"][0]))
        print("Metadata: ", len(result["metadatas"][0]))

        print("")

        #rank the videos and return the ranked videos.
        ranked_videos = self.ranker.rank(rag_results = result)

        print("Ranked videos: ", len(ranked_videos))

        return ranked_videos

class VideoQuery:
    
    def __init__(self):
        
        self.extractor = FrameExtractor()

        self.interpretor = FrameInterpreter()

        self.responder = VideoAssistant()
    
    def get_response(self, userquery, youtube_url, timestamp, context_seconds = 3, fps_sample = 2):

        context = self.extractor.build_video_context(
            youtube_url = youtube_url, 
            timestamp = timestamp, 
            context_seconds = context_seconds,
            fps_sample = fps_sample
            )
        
        frames = context['frames']

        question = self.interpretor.get_question(frames)

        interpretaion = self.interpretor.interpret_frames(question = question)

        answer = self.responder.answer_question(question = userquery, context = interpretaion)

        return answer



def test_feature_1():
    prompt = "Suggest me some CNN tutorials that are long form and rank them according to popularity"
    
    video_fetcher = RankedVideos()

    video_fetcher.get_ranked_videos(prompt = prompt)


def test_feature_2():

    user_Query = "what is the character name in the wallpaper?"
    youtube_url = "https://youtu.be/nVyD6THcvDQ"  # Example URL
    timestamp = 60.0

    vq = VideoQuery()
    answer = vq.get_response(userquery = user_Query, youtube_url = youtube_url, timestamp = timestamp)

    print(answer)

if __name__ == "__main__":

    # test_feature_1()
    test_feature_2()