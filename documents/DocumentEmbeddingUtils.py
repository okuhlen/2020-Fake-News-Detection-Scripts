from keras_preprocessing.text import text_to_word_sequence
from nltk import word_tokenize
from pandas import DataFrame
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import gensim.models.word2vec as w
import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

class DocumentEmbeddingUtils:

    def __init__(self, dataFrame: DataFrame):
        if dataFrame is None:
            raise Exception("A dataset must be passed")
        self.dataFrame = dataFrame

    def get_document_length(self, documents: []):
        if documents is None:
            raise Exception("Please provide a list of documents")

        max_length = 0
        longest_text = ""

        for i, row in documents.iterrows():
            if not isinstance(row["text"], str):
                continue

            word_tokens = word_tokenize(row["text"], language='english')
            if len(word_tokens) > max_length:
                max_length = len(word_tokens)
                longest_text = row["text"]

        print("longest document was " +str(max_length) + "words")
        return max_length

    def get_max_document_length(self, should_truncate: bool):
        """Returns a max count on the total documents in the dataset. Data preprocessing must have been done
        beforehand!"""

        max_length = 0
        longest_text = ""
        for index, row in self.dataFrame.iterrows():
            if not isinstance(row["text"], str):
                continue

            word_tokens = word_tokenize(row["text"], language="english")
            if len(word_tokens) > max_length:
                max_length = len(word_tokens)
                longest_text = row["text"]

        print("The longest document contains: " + str(max_length) + " words")
        print("The longest text:")
        #print(longest_text)
        if should_truncate:
            new_limit = int(max_length / 7)
            print("Word limit has been limited to " + str(new_limit) + " words")
            return new_limit

        return max_length

    def get_unique_word_count(self, word_count: int, documents: []):
        """This function assigns a number to each unique word appearing in the dataset (one-hot encoding)"""
        tokenizer = Tokenizer(num_words=word_count, oov_token="OOV")
        # might need to combine all column texts into one?
        if documents is None:
            raise Exception("Please supply a list of summarized documents")

        cleaned_document_list = []

        for i, row in documents.iterrows():
            word_tokens = text_to_word_sequence(row["text"], filters='!”#$%&()*+,-./:;<=>?@[\\]^_`{'
                                                                                  '|}~\t\n')
            if len(word_tokens) < word_count:
                cleaned_document_list.append(' '.join(word_tokens))
            else:
                cleaned_document_list.append(' '.join(word_tokens[0:word_count - 1]))

        tokenizer.fit_on_texts(texts=cleaned_document_list)
        encoded_words = tokenizer.texts_to_sequences(cleaned_document_list)
        return tokenizer.word_index, encoded_words

    def get_unique_words(self, word_count: int):
        """This function assigns a number to each unique word appearing in the dataset (one-hot encoding)"""
        tokenizer = Tokenizer(num_words=word_count, oov_token="OOV")
        # might need to combine all column texts into one?
        documentList = self.dataFrame["text"].tolist()

        cleaned_document_list = []

        for i in documentList:
            word_tokens = text_to_word_sequence(i, filters='!”#$%&()*+,-./:;<=>?@[\\]^_`{'
                                                                                  '|}~\t\n')
            if len(word_tokens) < word_count:
                cleaned_document_list.append(' '.join(word_tokens))
            else:
                cleaned_document_list.append(' '.join(word_tokens[0:word_count - 1]))

        tokenizer.fit_on_texts(texts=cleaned_document_list)
        encoded_words = tokenizer.texts_to_sequences(cleaned_document_list)

        print("Done computing list of unique words")
        return tokenizer.word_index, encoded_words

    def pad_corpus_words(self, max_word_count: int, encoded_dataset: []):
        """This function pads all entries shorter than the specified maximum length"""
        print("Now encoding and right-padding documents")
        padded_dataset = pad_sequences(encoded_dataset, max_word_count, padding='post')
        print("Document encoding and padding now complete")
        return padded_dataset

    def generate_summarized_text(self):
        device = torch.device('cuda')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')

        t5_model = T5ForConditionalGeneration.from_pretrained('t5-base').to(device)

        summarized_documents = []

        count = 1
        for index, row in self.dataFrame.iterrows():
            text = row["text"]
            if not isinstance(text, str):
                summarized_documents.append(" ")
                continue

            t5_text = "summarize: " + text
            tokenized_text = tokenizer.encode(t5_text, return_tensors="pt", max_length=512, truncation=True).to(device)

            summary_ids = t5_model.generate(tokenized_text,
                                         num_beams=4,
                                         no_repeat_ngram_size=2,
                                         min_length=20,
                                         max_length=200,
                                         early_stopping=True)

            output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            summarized_documents.append([output, row["label"]])
            print("Done summarizing article " + str(count))
            count = count + 1

        return summarized_documents


    def get_document_embeddings(self, word_dim, word_count, unique_word_list):
        """This method generates word embeddings based on the words in the vocabulary"""
        keylist = list(unique_word_list.keys())
        # previously used word_count.
        word_embeddings = np.zeros((len(keylist), word_dim))
        print("Now generating word embedding vectors")
        google_vectors = w.Word2VecKeyedVectors.load_word2vec_format("C:/Models/GoogleNews-vectors-negative300.bin",
                                                                     binary=True)
        print("Pre-Trained Word2Vec model loaded")
        index = 0
        totalNotInDictionary = 0
        invalid_words = []

        for word in keylist:
            try:
                wv = google_vectors[word]
                word_embeddings[index] = wv
                index = index + 1
            except KeyError:
                word_embeddings[index] = [0]
                totalNotInDictionary = totalNotInDictionary + 1
                invalid_words.append(word)

        print("Done generating word embeddings. A total of " + str(totalNotInDictionary) + " words were not in the "
                                                                                           "dictionary")

        return word_embeddings
