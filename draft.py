import pandas as pd
from sklearn.utils import shuffle
import sklearn
from sklearn.model_selection import train_test_split
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


#ytcomments_clean_aly = pd.read_excel('../offensivedetection/clean datasets/Subset2(Elizer)/subset2-aly-offensive.xlsx')
#print(ytcomments_clean_aly)

import langdetect as langd

"""
def language_detect(x):
    lang = langd.detect(x)
    return (lang)

ytcomments_clean_aly['l_detect'] = ytcomments_clean_aly['COMMENTS'].apply(language_detect)

print(type(ytcomments_clean_aly))

print(ytcomments_clean_aly.head())

lst = [('this is a test', 1), ('what language is this?', 4), ('stackoverflow is a website', 23)]
df = pd.DataFrame(lst, columns = ['text', 'something'])

#df['l_detect'] = df['text'].apply(language_detect)
#print(df.head())

"""

#--------------------------------------------------------------------------------------------- PRE PROCESSING

#pd.set_option('display.max_colwidth', None)

dataset = pd.read_excel('../offensivedetection/clean datasets/mergedset2.xlsx')

# Remove rows with NULL value
dataset = dataset.dropna().reset_index(drop=True)

# Convert to lowercase
dataset = dataset.apply(lambda x: x.astype(str).str.lower())

print(dataset.head())

# Balance data set
dataset_0 = dataset.loc[dataset['label'] == "y"][0:2]
dataset_1 = dataset.loc[dataset['label'] == "n"][0:2]

del dataset
dataset = pd.concat([dataset_0, dataset_1], ignore_index=True)
print(dataset['label'].value_counts())

# Just to shuffle
dataset = shuffle(dataset)
dataset_X = dataset[['comments']]
dataset_y = dataset['label']

#print(ytcomments_merged.info)

#for index, row in ytcomments_merged.iterrows():
  #if
   # print(row['COMMENTS'], row['OFFENSIVE'])

#print(ytcomments_merged.loc[ytcomments_merged['OFFENSIVE'] == 'Y', ['COMMENTS']])
#print(ytcomments_merged.loc[ytcomments_merged['COMMENTS'].str.contains("tanga", case=False)])
#print(type(ytcomments_merged))
#print(type(comments_only))

import nltk
from nltk.corpus import stopwords


#X = dataset.iloc[:, 0].values
#y = dataset.iloc[:, -1].values


#--------------------------------------------------------------------------------------------- DATA SPLIT

"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

X_train = X_train.tolist()
X_test = X_test.tolist()

category = ['y', 'n']

model.fit(X_train, y_train)

#--------------------------------------------------------------------------------------------- PREDICTING
#predicted_categories = model.predict(X_test)
#print(predicted_categories)
#print(X_test, [predicted_categories])

#custom function to have fun
def my_predictions(my_sentence, model):
    all_categories_names = np.array(category)
    prediction = model.predict([my_sentence])
    return prediction

my_sentence = "nakakadiri"
print("I predict: ", my_predictions(my_sentence, model))
"""




































