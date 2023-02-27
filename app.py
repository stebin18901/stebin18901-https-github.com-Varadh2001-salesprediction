import numpy as np 
import pandas as pd 
from sklearn.cluster import KMeans
import pickle

df = pd.read_csv("365_student_exams.csv")
x = df.iloc[:, [3, 4]].values

kmodel = KMeans(n_clusters=4, init='k-means++', random_state=0)
y_kmeans = kmodel.fit_predict(x)

# Save the KMeans model to a .pkl file
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmodel, f)
