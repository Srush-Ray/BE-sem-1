import numpy as np
from apyori import apriori
import pandas as pd

print("Apriori Algorithm")

data=pd.read_csv('Market_Basket_Optimisation.csv', header = None)

print("Data:")
print(data[:5])
print("Data Size : ",data.shape)

records=[]

for i in range(0,22):
	records.append([str(data.values[i,j]) for j in range(0,6)])

association_rules=apriori(records, min_support = 0.005, min_confidence = 0.5, min_lift = 3, min_length = 2)
association_result=list(association_rules)

print("Number of Association Rules Found : ",len(association_result))
print("First 10 Association Rules : ")
for i in range(0,10):
	print((i+1)," ",association_result[i].items)
