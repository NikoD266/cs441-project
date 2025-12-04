import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage



if __name__ == '__main__':
    little_data_df = pd.read_csv("little_data.csv")
    big_data_df = pd.read_csv("big_data.csv").set_index('player')

    features = [
        'aces_and_unret_percentage', 'df_percent', 'first_won_percent',
        'forced_per_point', 'unforced_per_point', 'winners_per_point',
        'net_frequency', 'net_points_won_ratio', 'net_winner_rate',
        'net_unforced_rate', 'net_forced_rate'
        # 'return_points_won_percent'
    ]

    X = big_data_df[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)



    n_clusters = 4
    link_type = "ward"

    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=link_type)
    labels = clustering.fit_predict(X_scaled)

    features_labeled  = X.copy()
    features_labeled['cluster'] = labels

    # print(features_labeled)

    for cl, group in features_labeled.groupby('cluster'):

        print(f"Cluster {cl} (n={len(group)}):", list(group.index[:10]))  # first 10 names
        print(features_labeled.loc[group.index])

    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("tab10", n_colors=len(np.unique(labels)))
    sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=labels, palette=palette, s=50, legend='full')
    plt.title(f'Agglomerative Clustering (n_clusters={n_clusters}) — PCA projection')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # 3d chart

    pca = PCA(n_components=3, random_state=42)
    proj = pca.fit_transform(X_scaled)
    plot_df = pd.DataFrame(proj, index=X.index, columns=['PC1','PC2','PC3'])

    plot_df['cluster'] = labels
    plot_df['player'] = plot_df.index
    print(plot_df)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(plot_df['PC1'], plot_df['PC2'], plot_df['PC3'],
               c=[palette[int(l) % len(palette)] for l in labels], s=40)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.show()

    # fig = px.scatter_3d(
    #     plot_df,
    #     x='PC1', y='PC2', z='PC3',
    #     color=plot_df['cluster'].astype(str),
    #     hover_name='player',
    #     hover_data={'cluster': True, 'PC1': False, 'PC2': False, 'PC3': False,
    #                 'player': True},
    #     labels={'color': 'cluster'},
    #     title='Interactive 3D PCA projection colored by cluster',
    #     width=900, height=700,
    #     color_discrete_sequence=px.colors.qualitative.T10
    # )
    # fig.show()


    # --- 6) Dendrogram (truncated)
    Z = linkage(X_scaled, method=link_type, metric='euclidean')
    p=10
    plt.figure(figsize=(10, 5))
    cut_height = Z[-(n_clusters - 1), 2]
    dendrogram(Z, truncate_mode='level', p=p, no_labels=True, color_threshold=cut_height)
    plt.title(f'Dendrogram (truncated to {p} last merges)')
    plt.xlabel('Samples (truncated)')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()

    plt.show()
