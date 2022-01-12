

from sklearn import cluster
import pandas as pd

df = pd.read_csv("./ch03/iris.data.csv")

df = df.iloc[:,0:4]

km = cluster.KMeans(n_clusters=3)
km.fit(df)

print(km.cluster_centers_)

