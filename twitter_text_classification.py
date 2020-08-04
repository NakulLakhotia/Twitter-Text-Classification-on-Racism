# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:36:25 2019

@author: nlakhotia
"""
'''  Load Dataset'''

import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report
dataset=pd.read_csv("twitter_text.csv")
#print(dataset.head(7))
 
''' Data Cleaning '''

#Clean text from noise
def clean_text(text):
    
    text = re.sub(r'[^a-zA-Z\']', ' ', text) #Filter to allow only alphabets
    text = re.sub(r'[^\x00-\x7F]+', '', text) #Remove Unicode characters
    text = text.lower() #Convert to lowercase to maintain consistency
    return text

dataset['clean_text'] = dataset.tweet.apply(lambda x: clean_text(x))
#print(dataset.head(7))

''' Feature Engineering '''

#Generate word frequency
def gen_freq(text):
    #Will store the list of words
    word_list = []

    #Loop over all the tweets and extract words into word_list
    for tw_words in text.split():
        word_list.extend(tw_words)

    #Create word frequencies using word_list
    word_freq = pd.Series(word_list).value_counts()
    
    #Drop the stopwords during the frequency calculation
    word_freq = word_freq.drop(stopwords.words('english'), errors='ignore')
    
    return word_freq

#Check whether a negation term is present in the text
def any_neg(words):
    for word in words:
        if word in ['n', 'no', 'non', 'not'] or re.search(r"\wn't", word):
            return 1
    else:
        return 0

#Check whether one of the 100 rare words is present in the text
def any_rare(words, rare_100):
    for word in words:
        if word in rare_100:
            return 1
    else:
        return 0

#Check whether prompt words are present
def is_question(words):
    for word in words:
        if word in ['when', 'what', 'how', 'why', 'who']:
            return 1
    else:
        return 0

word_freq = gen_freq(dataset.clean_text.str)
#100 most rare words in the dataset
rare_100 = word_freq[-100:]
#Number of words in a tweet
dataset['word_count'] = dataset.clean_text.str.split().apply(lambda x: len(x))
#Negation present or not
dataset['any_neg'] = dataset.clean_text.str.split().apply(lambda x: any_neg(x))
#Prompt present or not
dataset['is_question'] = dataset.clean_text.str.split().apply(lambda x: is_question(x))
#Any of the most 100 rare words present or not
dataset['any_rare'] = dataset.clean_text.str.split().apply(lambda x: any_rare(x, rare_100))
#Character count of the tweet
dataset['char_count'] = dataset.clean_text.apply(lambda x: len(x))

print(dataset.head(7))

#Top 10 common words are
print(word_freq[:10])
''' Splitting the dataset into Train-Test split - Machine Learning Model'''

X = dataset[['word_count', 'any_neg', 'any_rare', 'char_count', 'is_question']]
y = dataset.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=27)

''' Train an ML model for Text Classification '''

#Initialize GaussianNB classifier
model = GaussianNB()
#Fit the model on the train dataset
model = model.fit(X_train, y_train)
#Make predictions on the test dataset
pred = model.predict(X_test)

''' Evaluate ML Model '''
print("Accuracy:", accuracy_score(y_test, pred)*100, "%")
print()
print("Classification Report:\n",classification_report(y_test, pred))
