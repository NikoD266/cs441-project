import cluser_matrix as cm
import pandas as pd
import numpy as np



class model:
    # model type should be in ["gmm", "kmeans", "hierarchical", "elo"], anything else will fail to init
    # train list is a range of years (must be contiguous) to train on
    def __init__(self, model_type, train_list):
        if not model_type in ["gmm", "kmeans", "hierarchical", "elo"]:
            print("Bad model type, killing the whole process, shame on you")
            exit(2)

        self.cluster_probs = None
        self.model_type = model_type
        self.train_list = train_list

    # must run before win prob
    def train(self):
        if self.model_type == "elo":
            self.cluster_probs = np.ones((4,4)) * 0.5
            return
        clusters = pd.read_csv("clusters.csv")  # clusters dataframe
        matches = pd.read_csv("elo_output/elo_atp_matches_20xx.csv")  # matches dataframe (2000-present)
        mat = cm.get_cluster_matrix(clusters, matches, self.train_list[0],
                                    self.train_list[len(self.train_list) - 1], self.model_type).to_numpy()
        av_elo = cm.get_cluster_elo_matrices(clusters, matches, self.train_list[0],
                                             self.train_list[len(self.train_list) - 1], self.model_type)[0]
        elo_probs = np.zeros((4,4))
        for i in range(0,4):
            for j in range(0,4):
                elo_probs[i][j] = 1 / (1 + 10 ** ((av_elo[j][i] - av_elo[i][j]) / 400))

        # print(mat)
        # print(elo_probs)
        self.cluster_probs = np.zeros((4,4))
        for i in range(0,4):
            for j in range(0,4):
                thing1 = mat[i][j] / elo_probs[i][j]
                thing2 = mat[j][i] / elo_probs[j][i]
                self.cluster_probs[i][j] = thing1 / (thing1 + thing2)
        # print(self.cluster_probs)
        return self

    # will return the predicted win probability of player A winning
    #clusterA and clusterB are ints in [0, 1, 2, 3]
    def win_prob(self, eloA, clusterA, eloB, clusterB):
        if self.cluster_probs is None:
            print("Must train before running this method!")
            return None
        elo_prob = 1 / (1 + 10 ** ((eloB - eloA) / 400))
        cluster_prob = self.cluster_probs[clusterA][clusterB]
        # print(elo_prob)
        # print(cluster_prob)
        return elo_prob * cluster_prob / (elo_prob * cluster_prob + (1 - elo_prob) * (1 - cluster_prob))



# mod = model("gmm", range(2000, 2020))
# mod.train()
# print(mod.win_prob(1500, 1, 1800, 3))

