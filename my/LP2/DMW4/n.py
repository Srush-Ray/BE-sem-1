##!/usr/bin/env python3
## -*- coding: utf-8 -*-
#"""
#Created on Sun Apr 26 18:14:15 2020
#
#@author: srushti
#"""
#
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#
#dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t')
#
#import re
#import nltk
#nltk.download('stopwords')
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
#corpus=[]
#for i in range(0,1000) :
#    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i],)
#    review=review.lower()
#    review=review.split()
#    ps=PorterStemmer()
#    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
#    review=' '.join(review)
#    corpus.append(review)
#
#from sklearn.feature_extraction.text import CountVectorizer
#cv=CountVectorizer(max_features=1500)
#x=cv.fit_transform(corpus).toarray()
#y=dataset.iloc[:,1].values
#
#
#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#
#from sklearn.preprocessing import StandardScaler
#sc_X=StandardScaler()
#X_train=sc_X.fit_transform(X_train)
#X_test=sc_X.transform(X_test)
#
#from sklearn.naive_bayes import GaussianNB
#classifier=GaussianNB()
#classifier.fit(X_train,y_train)
#
#y_pred=classifier.predict(X_test)
#
#from sklearn.metrics import confusion_matrix
#cm=confusion_matrix(y_test,y_pred)
#
#
#
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

recall = np.diag(cm) / np.sum(cm, axis = 1)
precision = np.diag(cm) / np.sum(cm, axis = 0)

print('Precision of the model = ',precision)
print('Recall of the model = ',recall)








import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
recall = np.diag(cm) / np.sum(cm, axis = 1)
precision = np.diag(cm) / np.sum(cm, axis = 0)

print('Precision of the model = ',precision[0])
print('Recall of the model = ',recall[0])

























import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

#using dendogram for finding optimal number of clusterings
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Eucladian Distances')
plt.show()

#training the cluster Using HC
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)


plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()













import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

dataset=pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
print(dataset)

dataset.head()

dataset['Liked']=dataset['Liked'].map({1:1,0:2})

x = dataset['Review']
y = dataset['Liked']
print(x)
print(y)

vectorizer=TfidfVectorizer(stop_words = 'english')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.4, shuffle = False)
x_train_vec = vectorizer.fit_transform(x_train)
print(x_train_vec)
x_test_vec = vectorizer.transform(x_test)
nb = MultinomialNB()
nb.fit(x_train_vec, y_train)
nb.score(x_test_vec, y_test)
prediction = nb.predict(x_test_vec)
print(prediction.shape)
print(y_test.shape)
np.array(y_test)
print('Precision : ', precision_score(y_test, prediction, average = 'macro'))
print('Recall : ', recall_score(y_test, prediction, average = 'macro'))
print('F1 : ', f1_score(y_test, prediction, average = 'macro'))

