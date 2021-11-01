import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
import sklearn
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np

"""

ytcomments_aly = pd.read_excel('../offensivedetection/datasets/subset2-aly.xlsx')
ytcomments_aly_noduplicate = ytcomments_aly.drop_duplicates(subset=['COMMENTS'])

ytcomments_jas = pd.read_excel('../offensivedetection/datasets/subset2-jas.xlsx')
ytcomments_jas_noduplicate  = ytcomments_jas.drop_duplicates(subset=['COMMENTS'])

ytcomments_paul = pd.read_excel('../offensivedetection/datasets/subset2-paul.xlsx')
ytcomments_paul_noduplicate  = ytcomments_paul.drop_duplicates(subset=['COMMENTS'])

# CREATE EXCEL FILE
#ytcomments_aly_noduplicate.to_excel("subset2-aly-unique.xlsx", index=False)
#ytcomments_jas_noduplicate.to_excel("subset2-jas-unique.xlsx", index=False)
#ytcomments_paul_noduplicate.to_excel("subset2-paul-unique.xlsx", index=False)

# ALY
ytcomments_aly_drop = ytcomments_aly_noduplicate.drop(columns=['LANGUAGE (E, F, or N)', 'EMOTION',
       'Hard to annotate?\n(Y or N) ', 'Name o link? \n(Y, L or N) '])
#ytcomments_aly_drop.to_excel("subset2-aly-offensive.xlsx", index=False)

# JAS
ytcomments_jas_drop = ytcomments_jas_noduplicate.drop(columns=['LANGUAGE (E, F, or N)', 'EMOTION',
       'Hard to annotate?\n(Y or N) ', 'Name o link? \n(Y, L or N) '])
#ytcomments_jas_drop.to_excel("subset2-jas-offensive.xlsx", index=False)

# PAUL
ytcomments_paul_drop = ytcomments_paul_noduplicate.drop(columns=['LANGUAGE (E, F, or N)', 'EMOTION',
       'Hard to annotate?\n(Y or N) ', 'Name o link? \n(Y, L or N) '])
#ytcomments_paul_drop.to_excel("subset2-paul-offensive.xlsx", index=False)


#--------------------------------------------------------------------------------------------- FOR CECILIA SIVA
# ANGELA
ytcomments_angela = pd.read_excel('../offensivedetection/datasets/subset3-angela.xlsx')
ytcomments_angela_noduplicate = ytcomments_angela.drop_duplicates(subset=['COMMENTS'])

# JOHN
ytcomments_john = pd.read_excel('../offensivedetection/datasets/subset3-john.xlsx')
ytcomments_john_noduplicate = ytcomments_john.drop_duplicates(subset=['COMMENTS'])

# VERLYN
ytcomments_verlyn = pd.read_excel('../offensivedetection/datasets/subset3-verlyn.xlsx')
ytcomments_verlyn_noduplicate = ytcomments_verlyn.drop_duplicates(subset=['COMMENTS'])

# CREATE EXCEL FILE
#ytcomments_angela_noduplicate.to_excel("subset3-angela-unique.xlsx", index=False)
#ytcomments_john_noduplicate.to_excel("subset3-john-unique.xlsx", index=False)
#ytcomments_verlyn_noduplicate.to_excel("subset3-verlyn-unique.xlsx", index=False)

# ANGELA
ytcomments_angela_drop = ytcomments_angela_noduplicate.drop(columns=['LANGUAGE (E, F, or N)', 'EMOTION',
       'Hard to annotate?\n(Y or N) ', 'Name o link? \n(Y, L or N) '])
ytcomments_angela_drop.to_excel("subset3-angela-offensive.xlsx", index=False)

# JOHN
ytcomments_john_drop = ytcomments_john_noduplicate.drop(columns=['LANGUAGE (E, F, or N)', 'EMOTION',
       'Hard to annotate?\n(Y or N) ', 'Name o link? \n(Y, L or N) '])
ytcomments_john_drop.to_excel("subset3-john_-offensive.xlsx", index=False)


# VERLYN
ytcomments_verlyn_drop = ytcomments_verlyn_noduplicate.drop(columns=['LANGUAGE (E, F, or N)', 'EMOTION',
       'Hard to annotate?\n(Y or N) ', 'Name o link? \n(Y, L or N) '])
ytcomments_verlyn_drop.to_excel("subset3-verlyn-offensive.xlsx", index=False)

"""

#--------------------------------------------------------------------------------------------- PRE PROCESSING

#pd.set_option('display.max_colwidth', None)
dataset = pd.read_excel('../offensivedetection/clean datasets/mergedset4.xlsx')

# Remove rows with NULL value
dataset = dataset.dropna().reset_index(drop=True)

# Convert to lowercase
dataset = dataset.apply(lambda x: x.astype(str).str.lower())

# Balance data set
dataset_0 = dataset.loc[dataset['label'] == "y"][0:4]
dataset_1 = dataset.loc[dataset['label'] == "n"][0:4]

del dataset
dataset = pd.concat([dataset_0, dataset_1], ignore_index=True)
# Just to shuffle
dataset = shuffle(dataset)
dataset_X = dataset[['comments']]
dataset_y = dataset['label']

#--------------------------------------------------------------------------------------------- DATA SPLIT

X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_y, test_size=0.20, random_state=42)

#--------------------------------------------------------------------------------------------- FEATURE EXTRACTION METHODS

#Total number of WORDS in a document
def word_count_per_doc(text):
	tokenized = word_tokenize(cleaner(text))
	return len(tokenized)

def tokenize(text):
	tokenized = word_tokenize(cleaner(text))
	return (tokenized)

def cleaner(text):
	text = re.sub('[^a-zA-Z ]', '', str(text))
	text = re.sub(' +', ' ', str(text))
	cleaned_text = text.strip()
	return cleaned_text

def wordFrequency(wlist):
	wordlistinstring = ' '.join(wlist)

	wordfreq = []
	for w in wlist:
		wordfreq.append(wlist.count(w))

	# print("String\n" + wordlistinstring +"\n")
	# print("List\n" + str(wordlist) + "\n")
	# print("Frequencies\n" + str(wordfreq) + "\n")
	# print("Pairs\n" + str(list(zip(wordlist, wordfreq))))
	return wordfreq


#--------------------------------------------------------------------------------------------- Bar graph

# Remove column name to remove from count
dataset_X = dataset_X.rename(columns={"comments": ""})

# feature 1 - total number of offensive comments
f1 = len(dataset.loc[dataset['label'] == "y"])
# feature 2 - total number of normal comments
f2 = len(dataset.loc[dataset['label'] == "n"])
# feature 3 - total number of offensive and normal comments
f3 = len(dataset['comments'])

# feature 4 - total number of times a word appears in a normal comment
# feature 5 - total number of times a word appears in an offensive comment
# feature 6 - total number of words that appear in all of the normal comments
# feature 7 - total number of words that appear in all of the offensive comments
# feature 8 - probability of seeing each word given that it is in a normal comment
# feature 9 - probability of seeing each word given that it is in an offensive comment
# feature 10 - initial guess/prior probability that a comment is a normal comment
# feature 11 - initial guess/prior probability that a comment is an offensive comment

wc = word_count_per_doc(dataset_X)
wl = tokenize(dataset_X)
wf = wordFrequency(wl)

print("f1: ", f1)
print("f2: ", f2)
print("f3: ", f3)

# Total number of words in the dataset
print("Word Count:", wc)
print("List of Words: ", wl)
print("Frequency: ", wf)

x = wl;
y = wf
plt.bar(x,y)
plt.show()
































