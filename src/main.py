from prompt_preprocessor import PromptPreprocessor
from videoID_retrieval import VideoIDRetriever
from pprint import pprint
from frame_extractor import FrameExtractor
from video_frame_intrpreter import FrameInterpreter
from video_query_responder import VideoAssistant
from rag_to_url import RAGFetcher
from searching_ranking import VideoRanker
from helper.dataset import Dataset
import sys
from datetime import datetime

class RankedVideos:
    
    def __init__(self):
        
        self.VEC_DB_PATH = "Dataset\\video_index.faiss"
        self.df = Dataset.get_dataframe()

        print("DEBUG: LINE 18, DATAFRAME LOADED. \n")
        
        self.preprocessor = PromptPreprocessor()
        self.retriever = VideoIDRetriever(vecDB_path = self.VEC_DB_PATH, dataframe = self.df)
        self.ranker = VideoRanker()

        print("DEBUG: LINE 25, RankedVideos CONSTRUCTOR EXECUTED. \n")

    def get_ranked_videos(self, prompt: str):

        meta, vec = self.preprocessor.preprocess(prompt)

        print("")
        
        print("Prompt vector: ", vec.shape)

        results = self.retriever.get_videoIDs(meta["raw_prompt"], vec, k = 5)

        print("")

        print("Video IDs Retrieved: ", len(results.keys()))
        
        faiss_scores = [item[1]["faiss"] for item in results.items()]
        
        videoIDs = list(results.keys())
        
        #get the url
        self.url_fetcher = RAGFetcher(dataframe = self.df, faiss_scores = faiss_scores, videoIDs = videoIDs)
        result = self.url_fetcher.get_rag_results()

        #rank the videos and return the ranked videos.
        ranked_videos = self.ranker.rank(rag_results = result, weights = meta["weights"])

        self.ranker.display_results(ranked_videos)

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
        transcript = context['transcript']

        interpretaion = self.interpretor.interpret_frames(frames = frames, transcript = transcript)

        answer = self.responder.answer_question(question = userquery, context = interpretaion)

        return answer


def test_feature_1():
    
    print("Enter Prompt: " )
    prompt = sys.stdin.read()

    # prompt = "Give me a video about Pets & Animals and Pet Care"
    
    video_fetcher = RankedVideos()

    ranked = video_fetcher.get_ranked_videos(prompt = prompt)


def test_feature_2():

    # user_Query = "what is the character name in the wallpaper?"
    
    print("Enter Prompt: " )
    prompt = sys.stdin.read()

    print("Enter time like hours:minutes:seconds : ")
    timestamp = sys.stdin.read()

    time_str_ls = timestamp.split(":")

    time_str_ls = map(lambda x: x.strip(), time_str_ls)
    
    h, m, s = 0, 0, 0

    if len(list(time_str_ls)) == 3:

        h, m, s = map(int, time_str_ls)

        s += (h * 3600) + (m * 60)
    
    elif len(list(time_str_ls)) == 2:

        m, s = map(int, time_str_ls)

        s += m * 60
    
    elif len(list(time_str_ls)) == 1:

        s = map(int, time_str_ls)
    
    else:

        ValueError("Please enter the time in proper time foramt.")


    # youtube_url = "https://youtu.be/nVyD6THcvDQ"  # Example URL

    youtube_url = "https://www.youtube.com/watch?v=E_EQVLX_Kq0"

    timestamp = s

    vq = VideoQuery()
    answer = vq.get_response(userquery = prompt, youtube_url = youtube_url, timestamp = timestamp)

    pprint(answer)

if __name__ == "__main__":

    # test_feature_1()
    test_feature_2()