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
import emoji
from better_profanity import profanity
from sklearn.metrics import plot_confusion_matrix

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
#print(len(dataset))
#Remove rows with NULL value
dataset = dataset.dropna().reset_index(drop=True)

dataset = dataset.rename(columns={"COMMENTS": "comments", "OFFENSIVE (Y or N)": "label"})

#Convert to lowercase
dataset = dataset.apply(lambda x: x.astype(str).str.lower())

#--------------------------------------------------------------------------------------------- Balanced Data
#Balance data (Undersampling)

count_class_n, count_class_y = dataset.label.value_counts();

dataset_0 = dataset.loc[dataset['label'] == "y"]
dataset_1 = dataset.loc[dataset['label'] == "n"]

#UnderSampling
#print("Undersample")
dataset_1_under = dataset_1.sample(count_class_y)
dataset_bal = pd.concat([dataset_0, dataset_1_under], ignore_index=True)

#print(dataset_bal.shape)

#Balance data (Oversampling)
#print("Oversample")
#dataset_0_over = dataset_0.sample(count_class_n, replace=True)
#dataset_bal = pd.concat([dataset_0_over, dataset_1], ignore_index=True)

dataset_bal = shuffle(dataset_bal)
dataset_X = dataset_bal[['comments']]
dataset_y = dataset_bal['label']

X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(dataset_X, dataset_y, test_size=0.30, random_state=42)

X_train_bal.reset_index(drop=True, inplace=True)
X_test_bal.reset_index(drop=True, inplace=True)
y_train_bal.reset_index(drop=True, inplace=True)
y_test_bal.reset_index(drop=True, inplace=True)

#print("X_train :",X_train_bal.shape)
#print("X_test :",X_test_bal.shape)
#print("y_train :",y_train_bal.shape)
#print("y_test :",y_test_bal.shape)

"""
#--------------------------------------------------------------------------------------------- unbbalanced Data

# unbbalanced data
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
#print("X_train_unbal :",X_train_unbal.shape)
#print("X_test_unbal :",X_test_unbal.shape)
#print("y_train_unbal :",y_train_unbal.shape)
#print("y_test_unbal :",y_test_unbal.shape)
"""
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

"""
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
"""

def probability(wlist, total):
	mylist = []
	for i in range(len(wlist)):
		for j in range(len(wlist[i])):
			num = wlist[i][1]
			num = num / total

		mylist.append((wlist[i][0], num))
	return mylist

def wordFrequency(sentences):
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

def word_count(comment):
  return len(comment.split(' '))

def countChars(comment):
	return len(comment.replace(" ", "")) #do not include whitespace in character count

def has_emoji(comment):
		for character in comment:
			if character in emoji.UNICODE_EMOJI['en']:
				return True
		return False

def check_offensive_emojis(comment):
   emojis = ''.join(character for character in comment if character in emoji.UNICODE_EMOJI['en'])

   line = ["🖕", "💩", ""] #middle finger, pile of poo
   for character in emojis:
	   if character in line:
		   return 1
   return 0

def extract_offensive_emojis(comment):
	line = ["🖕", "💩", "🤬", "😠"] #middle finger, pile of poo, face with symbol on mouth, angry face
	return ''.join(character for character in comment if character in line)

def contains_english_swear_words(comment):
	if profanity.contains_profanity(comment):
		return 1
	else:
		return 0

def consonant_count(comment):
	comment = comment.lower()
	total_consonant = 0

	for i in comment:
		if i == 'b' or i == 'c' or i == 'd' or i == 'f' or i == 'g' \
				or i == 'h' or i == 'j' or i == 'k' or i == 'l' \
				or i == 'm' or i == 'n' or i == 'p' or i == 'q' \
				or i == 'r' or i == 's' or i == 't' or i == 'v' \
				or i == 'w' or i == 'x' or i == 'y' or i == 'z':
			total_consonant = total_consonant + 1;

	return total_consonant

def vowel_count(comment):
	comment = comment.lower()
	total_vowels = 0

	for i in comment:
		if i == 'a' or i == 'e' or i == 'i' or i == 'o' or i == 'u':
			total_vowels = total_vowels + 1;
	return total_vowels

def space_count(comment):
  return comment.count(' ')

def get_consonant_cluster(comment):
	cleaned = cleaner(comment)
	word_count = word_count_per_doc(comment)

	pattern = "([bcdfghjklmnpqrstvwxyz]{1}[bcdfghjklmnpqrstvwxyz]{1}[bcdfghjklmnpqrstvwxyz]*)"
	matches = len(re.findall(pattern, cleaned))

	if word_count > 0:
		return matches / word_count
	else:
		return 0

"""
#--------------------------------------------------------------------------------------------- Training Features (Unbalanced)
# feature 1 - Number of words
X_f1_unbal = X_train_unbal['comments'].apply(avg_words_per_sentence)
X_f1_unbal = X_f1_unbal.rename('f11')

vectorizer = CountVectorizer()
vectorizer.fit_transform(X_train_unbal['comments'])
fnames_train = vectorizer.get_feature_names_out()

# feature 2 - Word Frequency
X_f2_unbal = X_train_unbal['comments'].apply(wordFrequency)
X_f2_unbal = vectorizer.transform(X_train_unbal['comments'])
X_f2_unbal = pd.DataFrame(X_f2_unbal.toarray())

# feature 3 - Number of total characters in a comment
X_f3_unbal = X_train_unbal['comments'].apply(countChars)

# feature 4 - Check it the comment contains offensive emojis
X_f4_unbal = X_train_unbal['comments'].apply(check_offensive_emojis)

# feature 5 - If a comment contains an english swear word
X_f5_unbal = X_train_unbal['comments'].apply(contains_english_swear_words)

# feature 6 - Number of vowels in a comment
X_f6_unbal = X_train_unbal['comments'].apply(vowel_count)

# feature 7 - Number of consonants in a comment
X_f7_unbal = X_train_unbal['comments'].apply(consonant_count)

# feature 8 - Number of spaces in a comment
X_f8_unbal = X_train_unbal['comments'].apply(space_count)

# feature 9 - Consonant Count Density
X_f9_unbal = X_train_unbal['comments'].apply(get_consonant_cluster)

collected_features_unbal = pd.concat([X_f1_unbal, X_f2_unbal, X_f3_unbal, X_f4_unbal, X_f5_unbal, X_f6_unbal, X_f7_unbal, X_f8_unbal, X_f9_unbal], axis=1)
collected_features_unbal = collected_features_unbal.to_numpy();

y_train_unbal = y_train_unbal.to_numpy();
y_train_unbal = np.squeeze(y_train_unbal)
#--------------------------------------------------------------------------------------------- Test Features (Unbalanced)
#print("test: ", X_test_unbal)

# feature 1 - Number of words
X_f1_test_unbal = X_test_unbal['comments'].apply(avg_words_per_sentence)
X_f1_test_unbal = X_f1_test_unbal.rename('f11')

# feature 2 - Word Frequency
X_f2_test_unbal = X_test_unbal['comments'].apply(wordFrequency)
X_f2_test_unbal = vectorizer.transform(X_test_unbal['comments'])
X_f2_test_unbal = pd.DataFrame(X_f2_test_unbal.toarray())
fnames_test = vectorizer.get_feature_names_out()

# feature 3 - Number of total characters in a comment
X_f3_test_unbal = X_test_unbal['comments'].apply(countChars)

# feature 4 - Check it the comment contains offensive emojis
X_f4_test_unbal = X_test_unbal['comments'].apply(check_offensive_emojis)

# feature 5 - If a comment contains an english swear word
X_f5_test_unbal = X_test_unbal['comments'].apply(contains_english_swear_words)

# feature 6 - Number of vowels in a comment
X_f6_test_unbal = X_test_unbal['comments'].apply(vowel_count)

# feature 7 - Number of consonants in a comment
X_f7_test_unbal = X_test_unbal['comments'].apply(consonant_count)

# feature 8 - Number of spaces in a comment
X_f8_test_unbal = X_test_unbal['comments'].apply(space_count)

# feature 9 - Consonant Count Density
X_f9_test_unbal = X_test_unbal['comments'].apply(get_consonant_cluster)

collected_features_test_unbal = pd.concat([X_f1_test_unbal, X_f2_test_unbal, X_f3_test_unbal, X_f4_test_unbal, X_f5_test_unbal, X_f6_test_unbal, X_f7_test_unbal, X_f8_test_unbal, X_f9_test_unbal], axis=1)
"""
#--------------------------------------------------------------------------------------------- Training Features (Balanced)
# feature 1 - Number of words
X_f1_bal = X_train_bal['comments'].apply(avg_words_per_sentence)
X_f1_bal  = X_f1_bal .rename('f11')

vectorizer = CountVectorizer()
vectorizer.fit_transform(X_train_bal['comments'])
fnames_train = vectorizer.get_feature_names_out()

# feature 2 - Word Frequency
X_f2_bal = X_train_bal['comments'].apply(wordFrequency)
X_f2_bal = vectorizer.transform(X_train_bal['comments'])
X_f2_bal = pd.DataFrame(X_f2_bal.toarray())

# feature 3 - Number of total characters in a comment
X_f3_bal = X_train_bal['comments'].apply(countChars)

# feature 4 - Check it the comment contains offensive emojis
X_f4_bal = X_train_bal['comments'].apply(check_offensive_emojis)

# feature 5 - If a comment contains an english swear word
X_f5_bal = X_train_bal['comments'].apply(contains_english_swear_words)

# feature 6 - Number of vowels in a comment
X_f6_bal = X_train_bal['comments'].apply(vowel_count)
#print("X_f6:", X_f6)

# feature 7 - Number of consonants in a comment
X_f7_bal = X_train_bal['comments'].apply(consonant_count)
#print("X_f7:", X_f7)

# feature 8 - Number of spaces in a comment
X_f8_bal = X_train_bal['comments'].apply(space_count)

# feature 9 - Consonant Count Density
X_f9_bal = X_train_bal['comments'].apply(get_consonant_cluster)

collected_features_bal = pd.concat([X_f1_bal, X_f2_bal, X_f3_bal, X_f4_bal, X_f5_bal, X_f6_bal, X_f7_bal, X_f8_bal, X_f9_bal], axis=1)
collected_features_bal = collected_features_bal.to_numpy();

y_train_bal = y_train_bal.to_numpy();
y_train_bal = np.squeeze(y_train_bal)

#--------------------------------------------------------------------------------------------- Test Features (Balanced)
# feature 1 - Number of words
X_f1_test_bal = X_test_bal['comments'].apply(avg_words_per_sentence)
X_f1_test_bal = X_f1_test_bal.rename('f11')

# feature 2 - Word Frequency
X_f2_test_bal = X_test_bal['comments'].apply(wordFrequency)
X_f2_test_bal = vectorizer.transform(X_test_bal['comments'])
X_f2_test_bal = pd.DataFrame(X_f2_test_bal.toarray())
fnames_test = vectorizer.get_feature_names_out()

# feature 3 - Number of total characters in a comment
X_f3_test_bal = X_test_bal['comments'].apply(countChars)

# feature 4 - Check it the comment contains offensive emojis
X_f4_test_bal = X_test_bal['comments'].apply(check_offensive_emojis)

# feature 5 - If a comment contains an english swear word
X_f5_test_bal = X_test_bal['comments'].apply(contains_english_swear_words)

# feature 6 - Number of vowels in a comment
X_f6_test_bal = X_test_bal['comments'].apply(vowel_count)

# feature 7 - Number of consonants in a comment
X_f7_test_bal= X_test_bal['comments'].apply(consonant_count)

# feature 8 - Number of spaces in a comment
X_f8_test_bal = X_test_bal['comments'].apply(space_count)

# feature 9 - Consonant Count Density
X_f9_test_bal = X_test_bal['comments'].apply(get_consonant_cluster)

collected_features_test_bal = pd.concat([X_f1_test_bal, X_f2_test_bal, X_f3_test_bal, X_f4_test_bal, X_f5_test_bal, X_f6_test_bal, X_f7_test_bal, X_f8_test_bal, X_f9_test_bal], axis=1)
"""

#--------------------------------------------------------------------------------------------- Predict (Unbalanced)
mnb_unbal = MultinomialNB(alpha=1.0)
mnb_unbal.fit(collected_features_unbal, y_train_unbal)
y_pred_unbal = mnb_unbal.predict(collected_features_test_unbal)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print(classification_report(y_test_unbal, y_pred_unbal))

#--------------------------------------------------------------------------------------------- SVM Predict (Unbalanced)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
svclassifier = SVC(kernel='linear')
svclassifier.fit(collected_features_unbal, y_train_unbal)
y_pred_svm_unbal = svclassifier.predict(collected_features_test_unbal)

svc_gxboost = svclassifier.score(collected_features_test_unbal, y_test_unbal)
print('accuracy_gxboost: ', svc_gxboost)
print('f1 score: ', f1_score(y_test_unbal, y_pred_svm_unbal, average="macro"))
print('precision score: ', precision_score(y_test_unbal, y_pred_svm_unbal, average="macro"))
print('recall score', recall_score(y_test_unbal, y_pred_svm_unbal, average="macro"))

"""
#--------------------------------------------------------------------------------------------- MNB Predict (Balanced)
mnb_bal = MultinomialNB(alpha=1.0)
mnb_bal.fit(collected_features_bal, y_train_bal)

y_pred_bal = mnb_bal.predict(collected_features_test_bal)

print(classification_report(y_test_bal, y_pred_bal))
print(confusion_matrix(y_test_bal,y_pred_bal))

#--------------------------------------------------------------------------------------------- SVM Predict (Balanced)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
svclassifier = SVC(kernel='linear')
print('training svm')
svclassifier.fit(collected_features_bal, y_train_bal)
print('testing svm')
y_pred_svm_bal = svclassifier.predict(collected_features_test_bal)

svc_gxboost = svclassifier.score(collected_features_test_bal, y_test_bal)
print('accuracy_gxboost: ', svc_gxboost)
print('f1 score: ', f1_score(y_test_bal, y_pred_svm_bal, average="macro"))
print('precision score: ', precision_score(y_test_bal, y_pred_svm_bal, average="macro"))
print('recall score', recall_score(y_test_bal, y_pred_svm_bal, average="macro"))



















