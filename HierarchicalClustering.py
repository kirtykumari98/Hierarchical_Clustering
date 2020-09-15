#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the mail dataset with pandas
df=pd.read_csv("Mall_Customers.csv")
X=df.iloc[:,[3,4]].values

#using the dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(X,method='ward',metric='euclidean'))
plt.title("Dendogram")
plt.xlabel("customers")
plt.ylabel("distance")
plt.show()

#fitting hierarchical clustering to the mail dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage='ward')
y_hc=hc.fit_predict(X)

#visualising the clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=50,c='red')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=50,c='green')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=50,c='blue')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=50,c='magenta')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=50,c='cyan')
plt.title('Hierarchial Clustering')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score (1-100)')
plt.show()

