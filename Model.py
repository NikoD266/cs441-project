import cluser_matrix as cm
import pandas as pd
import numpy as np
import os



class model:
    def __init__(self, model_type, train_list, test_list):
        self.model_type = model_type
        self.train_list = train_list
        self.test_list = test_list

    def train(self):
        clusters = pd.read_csv("clusters.csv")  # clusters dataframe
        matches = pd.read_csv("tennis_atp/atp_matches_20xx.csv")  # matches dataframe (2000-present)
        mat = cm.get_cluster_matrix(clusters, matches, self.train_list[0], self.train_list[len(self.train_list) - 1], self.model_type)
        #av_elo = cm.get_cluster_elo_matrices(clusters, matches, self.train_list[0], self.train_list[len(self.train_list) - 1], self.model_type)
        av_elo = np.ones((4, 4)) * 1500

        print(mat)



mod = model("kmeans", range(2000, 2020), range(2020,2025))
mod.train()

