from prompt_preprocessor import PromptPreprocessor
from videoID_retrieval import VideoIDRetriever
from pprint import pprint
from frame_extractor import FrameExtractor
from video_frame_intrpreter import FrameInterpreter
from video_query_responder import VideoAssistant

class RankedVideos:
    
    def __init__(self):
        
        VEC_DB_PATH = "Dataset\\video_index.faiss"
        DATASET_PATH = "Dataset\\youtube_videos_dataset.csv"
    
        self.preprocessor = PromptPreprocessor()
        self.retriever = VideoIDRetriever(vecDB_path = VEC_DB_PATH, dataset_path = DATASET_PATH)

    def get_ranked_videos(self, prompt: str):

        meta, vec = self.preprocessor.preprocess(prompt)

        print("meta information of the prompt:")
        pprint(meta)

        print("Prompt vector: ", vec.shape)

        videoIDs, distances = self.retriever.get_videoIDs(vec, k = 5)

        print("Video IDs Retrieved: ", len(videoIDs))
        print("Distances of title and description of videos from the user prompt: ", len(distances))
        #get the url

        #rank the videos and return the ranked videos.

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



if __name__ == "__main__":

    prompt = "Suggest me some CNN tutorials that are long form and rank them according to popularity"
    
    video_fetcher = RankedVideos()

    video_fetcher.get_ranked_videos(prompt = prompt)