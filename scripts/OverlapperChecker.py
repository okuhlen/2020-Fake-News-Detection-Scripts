import pandas as pd
fake_news_file = pd.read_csv("../data/fake_news_vocabulary_1.txt", header=None)
real_news_file = pd.read_csv("../data/real_news_vocabulary_1.txt", header=None)

words_in_both_sets = []
counter = 0
for index, word in fake_news_file.iterrows():

    for inner_index, inner_word in real_news_file.iterrows():
        if not isinstance(word[0], str):
            continue

        if not isinstance(inner_word[0], str):
            continue

        if  word[0].lower() == inner_word[0].lower():
            words_in_both_sets.append(word)
            counter = counter + 1
            print(str(counter) + " words found in both sets")
        else:
            print("Done processing word number " +str(inner_index))

print("Overlapping word count: " +str(counter))
print("overlapping words")
print(words_in_both_sets)