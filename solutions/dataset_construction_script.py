import pandas as pd
import matplotlib as plt
import csv as c
fakeNewsArticles = []
conspiracyArticles = []

saDataFrame = pd.read_csv("C:/Projects/Research Projects/Master of Information "
                          + "Technology/Datasets/african_fake_news_db.csv", error_bad_lines=False, header=0)

maxLimit = 12000
count = 0
labels = ["fake", "satire", "bias", "conspiracy", "hate", "clickbait", "unreliable", "reliable",
          "political", "junksci"]
labelCount = [0,0,0,0,0,0,0,0,0,0]

chunkedDataFrame = []
grandTotal = maxLimit * len(labels)
c.field_size_limit(2147483647)

mergedDataFrame = pd.DataFrame()

for chunk in pd.read_csv("C:/Projects/Research Projects/Master of Information "
                              + "Technology/Datasets/OpenSources.csv",
                        chunksize=1500, skip_blank_lines=True, iterator=True, sep=",", header=0, error_bad_lines=False):

    chunkedDataFrame.append(chunk)

    sum = 0
    for i in labelCount:
        sum = sum + i
    if sum == grandTotal:
        break

    for index in chunkedDataFrame:

        for rowIndex, rowColumn in index.iterrows():

            sum = 0
            for i in labelCount:
                sum = sum + i
            if sum == grandTotal:
                break

            if str(rowColumn["type"]).strip() in labels:

                inputRow = pd.DataFrame([{'Title': str(rowColumn["title"]).strip(),
                             'Source': rowColumn["domain"],
                             'ArticleUrl': rowColumn["url"],
                             'DateAdded': rowColumn["scraped_at"],
                             'Content': rowColumn["content"],
                             'DateCreated': rowColumn["inserted_at"],
                             'Label': rowColumn["type"],
                             'LabelReason' : "",
                             'Keywords': rowColumn["keywords"]}])

                indexOfCol = labels.index(str(rowColumn["type"]).strip())

                if labelCount[indexOfCol] != maxLimit:
                    mergedDataFrame = pd.concat([mergedDataFrame, inputRow], ignore_index=True)
                    labelCount[indexOfCol] = labelCount[indexOfCol] + 1
            print(labelCount)

        sum = 0
        for i in labelCount:
            sum = sum + i
        if sum == grandTotal:
            break

print("Now merging original dataset")
print("Final dataset contains " + str(mergedDataFrame.shape[0]) + " records.")
mergedDataFrame.to_csv("C:/Projects/Research Projects/Master of Information "
                              "Technology/Datasets/custom_opensources_dataset.csv", sep=",", index=True, mode='w')
print("Merge complete!")







