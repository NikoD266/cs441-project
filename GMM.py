from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

df = pd.read_csv('little_data.csv')

df_view = df.iloc[:, [1, 3, 4, 5, 6, 7, 8, 9, 10]]
print(df_view.columns)

n = 2
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=67)
gmm.fit(df_view)
scores = []

# for i in range(1, 10):
#     gmm = GaussianMixture(n_components=i, covariance_type='full', random_state=67)
#     gmm.fit(df_view)
#     scores.append(gmm.aic(df_view))


print(gmm.means_)
print(gmm.weights_)
print(gmm.predict(df_view))
print(gmm.predict_proba(df_view))

pca = PCA(n_components=3)
pft = pca.fit_transform(df_view)
tsne = TSNE(n_components=3, random_state=528, perplexity=50)
xrt = tsne.fit_transform(df_view)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

print(list(df[gmm.predict(df_view) == 3]['player']))
sse = 0
for i in range(len(df_view)):
    cen = gmm.means_[gmm.predict(df_view)]
    sse += np.sum((np.array(df_view.iloc[i]) - cen) ** 2)

print(sse)

#ax.scatter(xrt[:, 0], xrt[:, 1], xrt[:, 2], c=gmm.predict(df_view), s=50, alpha=0.3)
#plt.title("GMM Cluster Visualization (with TSNE features)")
#plt.show()


# plt.plot(range(1, 10), scores)
# plt.xlabel("Number of components")
# plt.ylabel("AIC")
# plt.title("GMM Elbow Plot")
# plt.show()
