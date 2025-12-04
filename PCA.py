from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

lil = pd.read_csv("little_data.csv")
print(lil.head())
big_data = pd.read_csv("big_data.csv")
moderate_data = big_data.dropna(axis=1, how='any')
#print(moderate_data.columns)
smaller_data = moderate_data.iloc[:, [0, 1, 2, 4, 5, 6, 7, 8, 12, 16, 17, 18, 21, 22, 25, 26, 29, 30]]
#print(smaller_data.columns)
smallest_data = smaller_data.iloc[:, [3, 4, 5, 7, 8, 9, 10, 11, 12, 13]]
smallest_data = (smallest_data - smallest_data.mean()) / smallest_data.std()
smallest_data.insert(loc = 0, column = 'player', value = moderate_data['player'])
print(smallest_data.head())
smallest_data.to_csv("little_data.csv", index=False)
print(smallest_data.columns)
tsne = TSNE(n_components=2, random_state=52, perplexity=25)
xrt = tsne.fit_transform(smallest_data.iloc[:, 1:])
n=2
pca = PCA(n_components=n)
pft = pca.fit_transform(smallest_data.iloc[:, 1:])
for i in range(n):
    moderate_data.insert(loc=1, column = "PCA" + str(i), value=pft[:, i])

plt.scatter(pft[:, 0], pft[:, 1])
plt.show()
