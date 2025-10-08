import pandas as pd

df = pd.read_csv("./chartingData/charting-m-stats-ShotTypes.csv")

print(df)
df_net = df[df["row"] == "Net"].groupby("player", as_index=False).sum(numeric_only=True)
print(df_net)
df_total = df[df["row"] == "Total"].groupby("player", as_index=False).sum(numeric_only=True)
print(df_total)

df_net["net_winner_rate"] = df_net["winners"] / df_net["shots"]
df_net["net_unforced_rate"] = df_net["unforced"] / df_net["shots"]
df_net["net_forced_rate"] = df_net["induced_forced"] / df_net["shots"]
df_net["net_points_won_ratio"] = df_net["shots_in_pts_won"] / (df_net["shots_in_pts_won"] + df_net["shots_in_pts_lost"])

print(df_net)
df_merged = pd.merge(df_net, df_total, on="player", suffixes=("_net", "_total"))
df_merged["net_frequency"] = df_merged["shots_net"] / df_merged["shots_total"]

print(df_merged)
columns_to_keep = [
    "player",
    "net_frequency",
    "net_winner_rate",
    "net_unforced_rate",
    "net_forced_rate",
    "net_points_won_ratio",
]

df_summary = df_merged[columns_to_keep]
print(df_summary)
#df_summary.to_csv("net_summary.csv", index=False)
