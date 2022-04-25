import gensim as g
import pandas as pd
import nltk as tk
import numpy as np

#does this make sense?
doc2vecModel = g.models.doc2vec.Doc2Vec.load("../data/opensources_full_ds.model")

mergedDataset = pd.read_csv("../data/merged_fake_real_dataset.csv", chunksize=1000, error_bad_lines=False, sep=",")
featureDataFrame = pd.read_csv("../data/feature_set_july_20_2.csv", error_bad_lines=False)

counter = 0
print("Now attempting to generate document vectors on a pre-trained doc2vec model")

colCounter = 0
for chunk in mergedDataset:
    colCounter = 0
    for i, row in chunk.iterrows():

        if not isinstance(row["text"], str):
            vectors = np.zeros(400)
            for vec in vectors:
                if colCounter == 400:
                    break
                if "art_vect_" + str(colCounter) not in featureDataFrame.columns:
                    featureDataFrame["art_vect_" + str(colCounter)] = ""
                featureDataFrame.at[counter, "art_vect_" + str(colCounter)] = str(vec)
                print("Document vector " + str(colCounter) + " of document " + str(counter) + " added")
                colCounter = colCounter + 1

            counter = counter + 1
            colCounter = 0
            continue

        documentTokens = tk.word_tokenize(row["text"], language="english")
        vectors = doc2vecModel.infer_vector(documentTokens)
        colCounter = 0
        for vec in vectors:
            if colCounter == 400:
                break
            if "art_vect_" + str(colCounter) not in featureDataFrame.columns:
                featureDataFrame["art_vect_" + str(colCounter)] = ""
            featureDataFrame.at[counter, "art_vect_" + str(colCounter)] = str(vec)
            print("Document vector " + str(colCounter) + " of document " + str(counter) + " added")
            colCounter = colCounter + 1
        counter = counter + 1
        colCounter = 0

featureDataFrame.to_csv("../data/feature_set_with_doc_vectors_july_final.csv", header=True)