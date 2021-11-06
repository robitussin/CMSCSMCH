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
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

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
dataset = pd.read_excel('../offensivedetection/clean datasets/mergedset.xlsx')

#Remove rows with NULL value
dataset = dataset.dropna().reset_index(drop=True)

dataset = dataset.rename(columns={"COMMENTS": "comments", "OFFENSIVE (Y or N)": "label"})

#print("Label value counts: ", dataset['label'].value_counts())

#Convert to lowercase
dataset = dataset.apply(lambda x: x.astype(str).str.lower())

#--------------------------------------------------------------------------------------------- Balanced SPLIT
#Balance data
dataset_0 = dataset.loc[dataset['label'] == "y"][0:308]
dataset_1 = dataset.loc[dataset['label'] == "n"][0:308]

#del dataset
dataset_bal = pd.concat([dataset_0, dataset_1], ignore_index=True)
dataset_bal = pd.concat([dataset_1, dataset_0], ignore_index=True)

dataset_bal = shuffle(dataset_bal)
dataset_X = dataset_bal['comments']
dataset_y = dataset_bal['label']

X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(dataset_X, dataset_y, test_size=0.30, random_state=42)

X_train_bal.reset_index(drop=True, inplace=True)
X_test_bal.reset_index(drop=True, inplace=True)
y_train_bal.reset_index(drop=True, inplace=True)
y_test_bal.reset_index(drop=True, inplace=True)
#--------------------------------------------------------------------------------------------- UNBALANCED SPLIT

# Unbalance data
dataset_a = dataset.iloc[:, :-1].values
dataset_b = dataset.iloc[:, -1].values

dataset_a = pd.DataFrame(dataset_a, columns=['comments'])
dataset_b = pd.DataFrame(dataset_b)

X_train_unbal, X_test_unbal, y_train_unbal, y_test_unbal = train_test_split(dataset_a, dataset_b, test_size=0.30, random_state=42)


X_train_unbal.reset_index(drop=True, inplace=True)
X_test_unbal.reset_index(drop=True, inplace=True)
y_train_unbal.reset_index(drop=True, inplace=True)
y_test_unbal.reset_index(drop=True, inplace=True)
#---------------------------------------------------------------------------------------------
print("X_train :",X_train_unbal.shape)
print("X_test :",X_test_unbal.shape)
print("y_train :",y_train_unbal.shape)
print("y_test :",y_test_unbal.shape)

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
	# print("List\n" + str(wlist) + "\n")
	# print("Frequencies\n" + str(wordfreq) + "\n")
	#print("Pairs\n" + str(list(dict.fromkeys(zip(wlist, wordfreq)))))
	wlist = list(dict.fromkeys(zip(wlist, wordfreq)))
	#return wordfreq
	return wlist


def probability(wlist, total):
	#print("problist :", wlist)
	#print(type(wlist))
	mylist = []
	for i in range(len(wlist)):
		for j in range(len(wlist[i])):
			num = wlist[i][1]
			#print("num:", num)
			#print("total: ", total)
			num = num / total

		mylist.append((wlist[i][0], num))
	return mylist

def removestopwords(sentences):
	sentences = list(sentences)
	sentences = [word_tokenize(sentence) for sentence in sentences]
	for i in range(len(sentences)):
			sentences[i] = [word for word in sentences[i] if word not in stopwords.words('filipino')]
	return sentences


def avg_words_per_sentence(comment):
	sentences = comment.split('.')
	total_words = 0

	for sentence in sentences:
		total_words += word_count(sentence)

	return (total_words / len(sentences))

def word_count(article):
  return len(article.split(' '))
#--------------------------------------------------------------------------------------------- Bar graph

# Remove column name to remove from count
#dataset_X = dataset_X.rename(columns={"comments": ""})
#print(dataset_X)
#wc = word_count_per_doc(dataset_X)
#wl = tokenize(dataset_X)
#wf = wordFrequency(wl)

# Total number of words in the dataset
#print("Word Count:", wc)
#print("List of Words: ", wl)
#print("Frequency: ", wf)

#x = wl;
#y = wf
#plt.bar(x,y)
#plt.show()

# feature 1 - total number of offensive comments
f1 = len(dataset.loc[dataset['label'] == "y"])
# feature 2 - total number of normal comments
f2 = len(dataset.loc[dataset['label'] == "n"])
# feature 3 - total number of offensive and normal comments
f3 = len(dataset['comments'])

# feature 4 - total number of times a word appears in a normal comment
df = dataset.loc[dataset['label'] == "n"]
df = df[['comments']]
df = df.rename(columns={"comments": ""})
f4 = wordFrequency(tokenize(df))

# feature 5 - total number of times a word appears in an offensive comment
df2 = dataset.loc[dataset['label'] == "y"]
df2 = df2[['comments']]
df2 = df2.rename(columns={"comments": ""})
f5 = wordFrequency(tokenize(df2))

# feature 6 - total number of words that appear in all of the normal comments
f6 = len(tokenize(df))

# feature 7 - total number of words that appear in all of the offensive comments
f7 = len(tokenize(df2))

# feature 8 - probability of seeing each word given that it is in a normal comment

abc = f4
defh = f6
f8 = probability(abc, defh)

# feature 9 - probability of seeing each word given that it is in an offensive comment
abc = f5
defh = f7

f9 = probability(abc, defh)

# feature 10 - initial guess/prior probability that a comment is a normal comment
f10 = f2/(f2+f1)

# feature 11 - initial guess/prior probability that a comment is an offensive comment
f11 = f1/(f1+f2)

"""
print("f1: ", f1)
print("f2: ", f2)
print("f3: ", f3)
print("f4: ", f4)
print("f5: ", f5)
print(type(f5))
print("f6: ", f6)
print("f7: ", f7)
print("f8: ", f8)
print("f9: ", f9)
print("f10: ", f10)
print("f11: ", f11)
"""

#--------------------------------------------------------------------------------------------- Training Features
print("train: ", X_train_unbal)
X_f1 = X_train_unbal['comments'].apply(avg_words_per_sentence)
X_f1 = X_f1.rename('f11')

vectorizer = CountVectorizer()
vectorizer.fit_transform(X_train_unbal['comments'])
fnames_train = vectorizer.get_feature_names_out()

X_f2 = X_train_unbal['comments'].apply(removestopwords)
X_f2 = vectorizer.transform(X_train_unbal['comments'])
X_f2 = pd.DataFrame(X_f2.toarray())

collected_features_unbal = pd.concat([X_f1, X_f2], axis=1)
collected_features_unbal = collected_features_unbal.to_numpy();

y_train_unbal = y_train_unbal.to_numpy();
y_train_unbal = np.squeeze(y_train_unbal)
#--------------------------------------------------------------------------------------------- Test Features
print("test: ", X_test_unbal)

X_f1_test = X_test_unbal['comments'].apply(avg_words_per_sentence)
X_f1_test = X_f1_test.rename('f11')

X_f2_test = X_test_unbal['comments'].apply(removestopwords)
X_f2_test = vectorizer.transform(X_test_unbal['comments'])
X_f2_test = pd.DataFrame(X_f2_test.toarray())
fnames_test = vectorizer.get_feature_names_out()

collected_features_test_unbal = pd.concat([X_f1_test, X_f2_test], axis=1)

#--------------------------------------------------------------------------------------------- Predict
mnb = MultinomialNB(alpha=1.0)
mnb.fit(collected_features_unbal, y_train_unbal)

y_pred = mnb.predict(collected_features_test_unbal)

print("y_pred: ", y_pred)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print(classification_report(y_test_unbal, y_pred))

"""
# sample
simple_test = ['gago puta akin']
stpd = pd.DataFrame(data=simple_test, columns=['comments'])
stpd['comments'] = stpd[['comments']].apply(removestopwords)
simple_test2 = vectorizer.transform(simple_test)
print(simple_test2.toarray())
print(mnb.predict(simple_test2))

mnb = MultinomialNB(alpha=1.0)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sly.values.tolist())
print("X: ", X.toarray())
fnames = vectorizer.get_feature_names_out()

print(X.shape)
print(dataset_y.shape)
mnb.fit(X, dataset_y)


#example of predicting
simple_test = ['Lunch money money money money money']
simple_test2 = vectorizer.transform(simple_test)
print(simple_test2.toarray())
print(mnb.predict(simple_test2))


#example of removing stop words and prediction
simple_test = ['Lunch friend friend friend friend money akin']
stpd = pd.DataFrame(data=simple_test, columns=['comments'])
stpd['comments'] = stpd[['comments']].apply(removestopwords)
print("abc:", stpd)
simple_test2 = vectorizer.transform(simple_test)
print(mnb.predict(simple_test2))



#example of removing stop words and prediction
simple_test = ['pogi']
stpd = pd.DataFrame(data=simple_test, columns=['comments'])
stpd['comments'] = stpd[['comments']].apply(removestopwords)
print("abc:", stpd)
simple_test2 = vectorizer.transform(simple_test)
print(mnb.predict(simple_test2))
"""






















