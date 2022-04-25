import numpy as np
import pandas as pd

from PreProcessingUtils import PreProcessingUtils
from data_visualizations.DataVisualizationUtility import DataVisualizationUtility
from documents.DocumentEmbeddingUtils import DocumentEmbeddingUtils

"""Implicit Text Features: Basic pre-processing, create CNN model."""
documents = pd.read_csv("../data/merged_fake_real_dataset_sept.csv", chunksize=5000, error_bad_lines=False, sep=",",
                        )
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

rawDocumentLabels = np.asarray(cleaned_documents["label"].tolist())
deu = DocumentEmbeddingUtils(simplifiedDataFrame)

document_visualisation = DataVisualizationUtility(cleaned_documents)
document_visualisation.generate_word_cloud()