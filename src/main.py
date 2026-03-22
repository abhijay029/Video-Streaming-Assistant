from prompt_preprocessor import PromptPreprocessor
from videoID_retrieval import VideoIDRetriever
from pprint import pprint

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



if __name__ == "__main__":

    prompt = "Suggest me some CNN tutorials that are long form and rank them according to popularity"
    
    video_fetcher = RankedVideos()

    video_fetcher.get_ranked_videos(prompt = prompt)