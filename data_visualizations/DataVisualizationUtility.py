import pandas as pd
import wordcloud as wc
import matplotlib.pyplot as plt
from PIL import Image
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import io


class DataVisualizationUtility:

    def __init__(self, feature_data_frame: pd.DataFrame):
        if feature_data_frame is None:
            raise Exception("You need to supply a pre-processed data frame")
        self.data_frame = feature_data_frame

    def generate_word_cloud(self):

        fake_news_data = DataFrame()
        real_news_data = DataFrame()



        for index, row in self.data_frame.iterrows():
            if not isinstance(row["text"], str):
                continue

            if len(row["text"]) == 0:
                continue

            if row["label"] == 0:
                newRow = {"title": row["title"],
                          "text": row["text"],
                          "label": row["label"]}
                real_news_data = real_news_data.append(newRow, ignore_index=True)
            else:
                newRow = {"title": row["title"],
                          "text": row["text"],
                          "label": row["label"]}
                fake_news_data = fake_news_data.append(newRow, ignore_index=True)

        word_vectors = TfidfVectorizer(lowercase=True, analyzer="word", stop_words="english", max_features=None)
        fake_news_vectors = word_vectors.fit_transform(fake_news_data["text"])

        news_vectors2 = TfidfVectorizer(lowercase=True, analyzer="word", stop_words="english", max_features=None)
        real_news_vectors = news_vectors2.fit_transform(real_news_data["text"])

        fakeWordsFreq = {}
        realWordsFreq = {}

        with io.open("../data/fake_news_vocabulary_1.txt", "w+", encoding="utf-8") as f:
            for word, index in word_vectors.vocabulary_.items():
                sumOfColumn = fake_news_vectors.getcol(index).sum()
                fakeWordsFreq[word] = sumOfColumn
                f.write(word + "\n")

        with io.open("../data/real_news_vocabulary_1.txt", "w+", encoding="utf-8") as r:
            for word, index in news_vectors2.vocabulary_.items():
                sumOfCol = real_news_vectors.getcol(index).sum()
                realWordsFreq[word] = sumOfCol
                r.write(word + "\n")

        fake_word_cloud = WordCloud(width=1280, height=720, mode='RGBA', background_color='white',
                                    max_words=3000).fit_words(fakeWordsFreq)
        plt.imshow(fake_word_cloud)
        plt.axis("off")
        plt.show()
        plt.savefig("C:/outputs/fake_news_tags.png", format="png")

        real_word_cloud = WordCloud(width=1280, height=720, mode='RGBA', background_color='white',
                                    max_words=3000).fit_words(realWordsFreq)
        plt.imshow(real_word_cloud)
        plt.axis("off")
        plt.show()
        plt.savefig("C:/outputs/real_news_tags.png", format="png")
