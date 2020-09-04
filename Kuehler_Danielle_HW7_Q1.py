#Danielle Kuehler
#ITP 449 Summer 2020
#HW7
#Q1

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans

#1.Read the dataset into a dataframe. Be sure to import the header
wineDf = pd.read_csv("wineQualityReds.csv")
pd.set_option("Display.max_columns", None) #Display all columns

#2.Drop Wine from the dataframe
wineDf.drop("Wine", axis=1, inplace=True)

#3.Extract Quality and store it in a separate variable
quality = wineDf["quality"]

#4.Drop Quality from dataframe
wineDf.drop("quality", axis=1, inplace=True)

#5.Print the dataframe and Quality
print("5. Wine dataframe:\n", wineDf)
print("\n5. Quality:\n", quality)

#6.Normalize all columns of the dataframe. Use the Normalizer class from sklearn.preprocessing
n = Normalizer()
n.fit(wineDf) #Fit it to wine dataset- normalize every column in wine data set
wineDf_norm = pd.DataFrame(n.transform(wineDf), columns=wineDf.columns) #Use transform function of object n, convert fitted data into dataframe

#7.Print the normalized dataframe
print("\n7. Normalized wine dataframe:\n", wineDf_norm)

#8.	Create a range of k values from 1:11 for KMeans clustering. Iterate on the k values and store the inertia for each clustering in a list
ks = range(1,11) #Range of k values
inertias = [] #Initialize intertia list

#Every value of K, compute inertia
for k in ks:
    model = KMeans(n_clusters=k, random_state=10) #k in range 1:11
    model.fit(wineDf_norm) #Considers all attributes in dataframe to create k number of clusters
    inertias.append(model.inertia_) #Append to inertia list

#9.Plot the chart of inertia vs number of clusters
plt.plot(ks, inertias, "-o") #X axis is number of clusters, Y axis is difference within clusters
#Formatting
plt.xlabel("Number of Clusters, K")
plt.ylabel("Intertia")
plt.show() #Display graph

#10. I would pick 6 clusters for KMeans

#11. Now cluster the wines into K clusters. Assign the respective cluster number to each wine. Print the dataframe showing the cluster number for each wine
model = KMeans(n_clusters=6, random_state=10) #Cluster wines into 5 clusters with random state set to 10 for consistency
model.fit(wineDf_norm) #Fit each wine to cluster considering attributes of dataframe

labels = model.predict(wineDf_norm) #To generate cluster labels
wineDf_norm["Cluster"] = pd.Series(labels) #Add new column to dataframe that shows each wine's cluster assignment
print("11. Wine Dataframe with clusters:\n", wineDf_norm) #Display

#12. Add quality back to dataframe
#Did not want to consider cluster assignment based on quality, now put data back
wineDf_norm["quality"] = pd.Series(quality)
print("12. Wine Dataframe with clusters and quality\n", wineDf_norm)

#13. Now print a crosstab (from Pandas) of cluster number vs quality. Comment if the clusters represent the quality of wine
print(pd.crosstab(index=wineDf_norm['quality'], columns=wineDf_norm['Cluster']))
'''The clusters do not represent the quality of wine. Not only was the quality column removed when determining which wine belonged to each cluster, 
but also in the crosstab there is no one cluster with the highest or lowest quality consistently for all cluster members. Each cluster has a variety of
wine quality levels'''



