import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as mlt

# Load the student exam data
df = pd.read_csv("365_student_exams.csv")
x = df.iloc[:, [3, 4]].values

# Train the KMeans model
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++', random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
kmodel = KMeans(n_clusters=4, init='k-means++', random_state=0)
y_kmeans = kmodel.fit_predict(x)

# Save the KMeans model to a .pkl file
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmodel, f)

# Define the Streamlit app
st.title('Student Exam Clustering')

# Get the exam result from the user
result = st.slider('Enter the exam result:', 0.0, 100.0, step=0.1)
result2= st.slider('Enter the exam completion time:', 0.0, 100.0, step=0.1)

# Create a 2D array with the exam result
data = np.array([[result, result2]])


# Load the KMeans model from the saved .pkl file
with open('kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

# Use the KMeans model to predict the cluster label for the exam result
cluster_label =kmeans_model.predict(data)

# Display the predicted cluster label to the user
st.write('The exam result belongs to Cluster', cluster_label[0]+1)

# Plot the clustering
mlt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=80,c="red",label='Cluster 1')
mlt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=80,c="blue",label='Cluster 2')
mlt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=80,c="yellow",label='Cluster 3')
mlt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=80,c="cyan",label='Cluster 4')
mlt.scatter(kmeans_model.cluster_centers_[:,0],kmeans_model.cluster_centers_[:,1],s=100,c='magenta',label='Centroids')
mlt.title('Student Exam Clustering')
mlt.xlabel('Result')
mlt.ylabel('Time')
st.pyplot(mlt)
