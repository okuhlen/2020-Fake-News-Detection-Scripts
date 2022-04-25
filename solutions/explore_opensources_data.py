import pandas as pd
import numpy as np
import nltk as nl
import sklearn as sc
import csv as c
import os as o

c.field_size_limit(2147483647)

openSourcesDataSet = pd.read_csv("C:/Projects/Research Projects/Master of Information "
                               "Technology/Datasets/OpenSources.csv",
                         sep=",", header=0, iterator=True, error_bad_lines=False,
                        skiprows=0, skip_blank_lines=True, nrows=10000)

dataframe = pd.DataFrame(openSourcesDataSet)
print(dataframe.head())







