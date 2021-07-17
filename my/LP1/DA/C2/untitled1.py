import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

data=pd.read_csv('Pima.csv')
data.head(5)
data.shape
data.info()

data['x1'].describe() 

data.dtypes  

train=np.array(data.iloc[0:600])  #0 to 600th rows and all cols
test=np.array(data.iloc[600:768])   #600 to 768th rows and all cols

train.shape
test.shape

model = GaussianNB() 
Y=train[:,8]
print(Y)
model.fit(train[:,0:8], train[:,8])     #dependent and independ. variable
predicted= model.predict(test[:,0:8])
print(test[:,8])
print(predicted)

count=0
for l in range(168):
    if(predicted[l]==test[l,8]):
        count=count+1

print("Matched samples:",count)

print("Accuracy:",(count/168))


