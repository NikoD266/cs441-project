import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

clusters = pd.read_csv("C:\\Users\\ejpot\\Downloads\\clusters.csv") # clusters dataframe
matches = pd.read_csv("C:\\Users\\ejpot\\Downloads\\elo_atp_matches_20xx.csv") # matches dataframe (2000-present)

def get_cluster_matrix(clusters_data, matches_data, year_start, year_end, cluster_type):

    # clusters_data = pandas dataframe with clusters assignments
    # matches_data = pandas dataframe with match history
    # year_start = integer of starting year
    # year_end = integer of ending year
    # cluster_type = either "gmm", "hierarchical", or "kmeans"

    clusters_data = clusters_data.copy()
    cluster_cols = ["GMM_Cluster", "Kmeans_cluster", "hierarchial_cluster"]

    for col in cluster_cols:
        clusters_data[col] = clusters_data[col].astype("Int64")

    cluster_col_map = {
        "gmm": "GMM_Cluster",
        "kmeans": "Kmeans_cluster",
        "hierarchical": "hierarchial_cluster"
    }
    if cluster_type not in cluster_col_map:
        raise ValueError("cluster_type must be gmm, kmeans, or hierarchical")

    cluster_col = cluster_col_map[cluster_type]

    matches_data = matches_data.copy()
    matches_data["year"] = matches_data["tourney_date"].astype(str).str[:4].astype(int)

    matches = matches_data.loc[
        (matches_data["year"] >= year_start) &
        (matches_data["year"] <= year_end),
        ["tourney_date", "winner_name", "loser_name"]
    ].copy()

    cluster_lookup = clusters_data[["player", cluster_col]].rename(
        columns={cluster_col: "cluster"}
    )

    matches = matches.merge(
        cluster_lookup, how="left",
        left_on="winner_name", right_on="player"
    ).rename(columns={"cluster": "winner_cluster"}).drop(columns=["player"])

    matches = matches.merge(
        cluster_lookup, how="left",
        left_on="loser_name", right_on="player"
    ).rename(columns={"cluster": "loser_cluster"}).drop(columns=["player"])

    matches["null_count"] = matches[["winner_cluster", "loser_cluster"]].isna().sum(axis=1)
    print(f"Rows with 0 missing clusters: {(matches['null_count'] == 0).sum()}")
    print(f"Rows with 1 missing cluster:  {(matches['null_count'] == 1).sum()}")
    print(f"Rows with 2 missing clusters: {(matches['null_count'] == 2).sum()}")
    print()

    df = matches.dropna(subset=["winner_cluster", "loser_cluster"]).copy()

    pair_counts = (
        df.groupby(["winner_cluster", "loser_cluster"])
          .size()
          .reset_index(name="wins")
    )

    win_counts = pair_counts.pivot_table(
        index="winner_cluster",
        columns="loser_cluster",
        values="wins",
        fill_value=0
    )

    clusters = sorted(df["winner_cluster"].unique())
    win_counts = win_counts.reindex(index=clusters, columns=clusters, fill_value=0)

    win_counts = win_counts.astype("Int64")
    match_counts = win_counts + win_counts.T
    match_counts = match_counts.astype("Int64")
    match_counts = win_counts + win_counts.T

    win_pct = win_counts / match_counts.replace(0, pd.NA)

    print("Win counts:")
    print(win_counts, "\n")

    print("Match counts:")
    print(match_counts, "\n")

    print("Win percentages:")
    print(win_pct)

    return ""



def get_cluster_elo_matrices(clusters_data, matches_data, year_start, year_end, cluster_type):

    clusters_data = clusters_data.copy()
    cluster_cols = ["GMM_Cluster", "Kmeans_cluster", "hierarchial_cluster"]

    for col in cluster_cols:
        clusters_data[col] = clusters_data[col].astype("Int64")

    cluster_col_map = {
        "gmm": "GMM_Cluster",
        "kmeans": "Kmeans_cluster",
        "hierarchical": "hierarchial_cluster"
    }
    if cluster_type not in cluster_col_map:
        raise ValueError("cluster_type must be gmm, kmeans, or hierarchical")

    cluster_col = cluster_col_map[cluster_type]

    matches_data = matches_data.copy()
    matches_data["year"] = matches_data["tourney_date"].astype(str).str[:4].astype(int)

    matches = matches_data.loc[
        (matches_data["year"] >= year_start) &
        (matches_data["year"] <= year_end),
        ["winner_name", "loser_name", "r_w_global_before", "r_l_global_before",
         "r_w_surface_before", "r_l_surface_before"]
    ].copy()

    cluster_lookup = clusters_data[["player", cluster_col]].rename(columns={cluster_col: "cluster"})

    # Merge winner clusters
    matches = matches.merge(
        cluster_lookup, how="left", left_on="winner_name", right_on="player"
    ).rename(columns={"cluster": "winner_cluster"}).drop(columns=["player"])

    # Merge loser clusters
    matches = matches.merge(
        cluster_lookup, how="left", left_on="loser_name", right_on="player"
    ).rename(columns={"cluster": "loser_cluster"}).drop(columns=["player"])

    # Drop rows with missing clusters
    df = matches.dropna(subset=["winner_cluster", "loser_cluster"]).copy()

    clusters = sorted(df["winner_cluster"].unique())

    # Function to compute combined average ELO
    def compute_avg_elo(df, elo_col_winner, elo_col_loser):
        # Average winner ELO for cluster_i vs cluster_j
        df1 = df.groupby(["winner_cluster", "loser_cluster"])[elo_col_winner].mean().reset_index()
        # Average loser ELO for cluster_j vs cluster_i
        df2 = df.groupby(["loser_cluster", "winner_cluster"])[elo_col_loser].mean().reset_index()
        df2 = df2.rename(columns={
            "loser_cluster": "winner_cluster",
            "winner_cluster": "loser_cluster",
            elo_col_loser: elo_col_winner
        })
        # Combine
        combined = pd.concat([df1, df2])
        matrix = combined.groupby(["winner_cluster", "loser_cluster"])[elo_col_winner].mean().unstack(fill_value=pd.NA)
        matrix = matrix.reindex(index=clusters, columns=clusters, fill_value=pd.NA)
        return matrix

    avg_global_elo = compute_avg_elo(df, "r_w_global_before", "r_l_global_before")
    avg_surface_elo = compute_avg_elo(df, "r_w_surface_before", "r_l_surface_before")

    print("Average Global ELO:")
    print(avg_global_elo, "\n")

    print("Average Surface ELO:")
    print(avg_surface_elo, "\n")

    return avg_global_elo, avg_surface_elo




get_cluster_matrix(clusters, matches, 2000, 2025, "kmeans")
get_cluster_elo_matrices(clusters, matches, 2000, 2025, "kmeans")



# RESULTS OF ALL MATCHES FOR ALL YEARS:



# GMM (2000-2025):

# Rows with 0 missing clusters: 55658
# Rows with 1 missing cluster:  14879
# Rows with 2 missing clusters: 4369

# Win counts:
# loser_cluster      0      1     2     3
# winner_cluster
# 0               3130   3952  1059   794
# 1               8210  25180  3168  3036
# 2               1302   2031   520   310
# 3                893   1553   226   294

# Match counts:
# loser_cluster       0      1     2     3
# winner_cluster
# 0                6260  12162  2361  1687
# 1               12162  50360  5199  4589
# 2                2361   5199  1040   536
# 3                1687   4589   536   588

# Win percentages:
# loser_cluster          0         1         2         3
# winner_cluster
# 0                    0.5  0.324947  0.448539  0.470658
# 1               0.675053       0.5  0.609348  0.661582
# 2               0.551461  0.390652       0.5  0.578358
# 3               0.529342  0.338418  0.421642       0.5


# HIERARCHICAL (2000-2025):

# Rows with 0 missing clusters: 55658
# Rows with 1 missing cluster:  14879
# Rows with 2 missing clusters: 4369

# Win counts:
# loser_cluster       0    1     2     3
# winner_cluster
# 0               20594  401  8619  1207
# 1                 256    5    82    16
# 2               13845  224  8082   645
# 3                1095   27   419   141

# Match counts:
# loser_cluster       0    1      2     3
# winner_cluster
# 0               41188  657  22464  2302
# 1                 657   10    306    43
# 2               22464  306  16164  1064
# 3                2302   43   1064   282

# Win percentages:
# loser_cluster          0         1         2         3
# winner_cluster
# 0                    0.5   0.61035  0.383681  0.524327
# 1                0.38965       0.5  0.267974  0.372093
# 2               0.616319  0.732026       0.5  0.606203
# 3               0.475673  0.627907  0.393797       0.5



# K-MEANS (2000-2025):

# Rows with 0 missing clusters: 55658
# Rows with 1 missing cluster:  14879
# Rows with 2 missing clusters: 4369

# Win counts:
# loser_cluster      0      1     2     3
# winner_cluster
# 0               3907   5233  1302  2357
# 1               7509  14239  2592  5115
# 2               1344   1873   509   749
# 3               2467   3863   695  1904

# Match counts:
# loser_cluster       0      1     2     3
# winner_cluster
# 0                7814  12742  2646  4824
# 1               12742  28478  4465  8978
# 2                2646   4465  1018  1444
# 3                4824   8978  1444  3808

# Win percentages:
# loser_cluster          0         1         2         3
# winner_cluster
# 0                    0.5  0.410689  0.492063  0.488599
# 1               0.589311       0.5  0.580515  0.569726
# 2               0.507937  0.419485       0.5  0.518698
# 3               0.511401  0.430274  0.481302       0.5
