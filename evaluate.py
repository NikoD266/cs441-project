import numpy as np
from Model import model
import pandas as pd


def evaluate_brier_score(model_obj, testing_matches, clusters_df, cluster_col_name):
    cluster_lookup = clusters_df[["player", cluster_col_name]].rename(columns={cluster_col_name: "cluster"})
    df = testing_matches.copy()
    df = df.merge(cluster_lookup, how="left", left_on="winner_name", right_on="player").rename(
        columns={"cluster": "winner_cluster"}).drop(columns=["player"])
    df = df.merge(cluster_lookup, how="left", left_on="loser_name", right_on="player").rename(
        columns={"cluster": "loser_cluster"}).drop(columns=["player"])
    df = df.dropna(subset=["winner_cluster", "loser_cluster"])

    df["winner_cluster"] = df["winner_cluster"].astype(int)
    df["loser_cluster"] = df["loser_cluster"].astype(int)

    preds = []
    actuals = []

    for _, row in df.iterrows():
        eloA = row["r_w_global_before"]
        eloB = row["r_l_global_before"]
        clusterA = row["winner_cluster"]
        clusterB = row["loser_cluster"]
        # predicted probability that *winner* wins
        p = model_obj.win_prob(eloA, clusterA, eloB, clusterB)

        preds.append(p)
        actuals.append(1)  # winner will always be 1

    preds = np.array(preds)
    actuals = np.array(actuals)
    brier = np.mean((preds - actuals) ** 2)
    return brier


m = model("gmm", train_list=range(2000, 2015))
m.train()

test_matches = pd.read_csv("Elo_Output/elo_history.csv")
clusters = pd.read_csv("clusters.csv")
print("Brier Score:", evaluate_brier_score(m,test_matches,clusters,"GMM_Cluster"))
