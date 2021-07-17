#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np


# In[3]:


data = pd.read_csv("bbc-text.csv")
data.head()


# In[4]:


data['category'] = data['category'].map({'tech' : 1, 'business' : 2, 'sport' : 3, 'entertainment' : 4, 'politics' : 5})
x = data['text']
y = data['category']

print(y)
# In[5]:


vectorizer = TfidfVectorizer(stop_words = 'english')
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, shuffle = False)
x_train_vec = vectorizer.fit_transform(x_train)
print(x_train_vec)


# In[6]:


x_test_vec = vectorizer.transform(x_test)
nb = MultinomialNB()
nb.fit(x_train_vec, y_train)


# In[7]:


nb.score(x_test_vec, y_test)


# In[8]:


prediction = nb.predict(x_test_vec)


# In[9]:


print(prediction.shape)
print(y_test.shape)


# In[10]:


np.array(y_test)


# In[11]:


print('Precision : ', precision_score(y_test, prediction, average = 'macro'))
print('Recall : ', recall_score(y_test, prediction, average = 'macro'))
print('F1 : ', f1_score(y_test, prediction, average = 'macro'))


# In[ ]:




