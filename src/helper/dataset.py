import pandas as pd
import os

class Dataset:

    @staticmethod
    def get_dataframe():

        path = "Dataset\\youtube_videos_dataset.csv"

        if not os.path.exists(path):

            raise FileNotFoundError("Dataset not found: It may not exists \n Path: ", path)
                
        return pd.read_csv(path, parse_dates=['published_at'])

if __name__ == "__main__":

    dataset = Dataset.get_dataframe()

    for _, row in dataset.iterrows():

        print(row['published_at'].year)
        print(type(row['published_at']))

        break