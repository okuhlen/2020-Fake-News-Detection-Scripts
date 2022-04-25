import pandas as pd
import numpy as np
from PreProcessingUtils import PreProcessingUtils
from documents.DocumentEmbeddingUtils import DocumentEmbeddingUtils

"""The purpose of this script is to summarize text contained in the corpora"""

documents = pd.read_csv("../data/merged_fake_real_dataset_sept.csv",
                        chunksize=5000,
                        error_bad_lines=False,
                        sep=",")

special_characters = pd.read_csv("../data/special_characters.txt", sep="-", header=None, low_memory=True)
cleaned_documents = pd.DataFrame()
simplifiedDataFrame = pd.DataFrame()
for i in documents:
    pp = PreProcessingUtils(i, ["title", "text"])
    pp.remove_stop_words()
    pp.remove_special_characters(special_characters)
    temp = pp.get_data_frame()
    simplifiedDataFrame = pd.concat([simplifiedDataFrame, pp.concatenate_title_and_body()])
    cleaned_documents = pd.concat([cleaned_documents, pp.get_data_frame()])

#rawDocumentLabels = np.asarray(cleaned_documents["label"].tolist())
deu = DocumentEmbeddingUtils(cleaned_documents)
summarized_documents = deu.generate_summarized_text()

df = pd.DataFrame(summarized_documents)
df.to_csv(path_or_buf="../data/summarized_text_11.csv", sep=',', mode='w')

