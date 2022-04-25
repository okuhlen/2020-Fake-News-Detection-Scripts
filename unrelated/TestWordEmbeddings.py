import gensim as g
import pandas as pd
import nltk as tk
import numpy as np

doc2vecModel = g.models.doc2vec.Doc2Vec.load("../data/opensources_full_ds.model")
print("Words similiar to Korea:")
#noinspection
print(doc2vecModel.wv.similar_by_word("Korea"))