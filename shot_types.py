import pandas as pd

data = pd.read_csv("C:\\Users\\ejpot\\Downloads\\charting-m-stats-ShotTypes.csv")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)

data = data[data["match_id"].astype(str).str.startswith("2")]

shots_pivot = data.pivot_table(
    index="player",
    columns="row",
    values="shots",
    aggfunc="sum",
    fill_value=0
).reset_index()

pivot_extended = data.pivot_table(
    index="player",
    columns="row",
    values=["shots", "winners", "induced_forced", "unforced"],
    aggfunc="sum",
    fill_value=0
)

pivot_extended.columns = [f"{shot}_{val}" for val, shot in pivot_extended.columns]
pivot_extended = pivot_extended.reset_index()

cols = pivot_extended.columns.tolist()
cols_sorted = ['player'] + sorted([col for col in cols if col != 'player'])
pivot_extended = pivot_extended[cols_sorted]

shot_types = set(col.split('_')[1] for col in pivot_extended.columns if col.startswith("shots_"))

for shot in shot_types:
    shots_col = f"shots_{shot}"
    winners_col = f"winners_{shot}"
    induced_col = f"induced_forced_{shot}"
    unforced_col = f"unforced_{shot}"
    other_col = f"{shot}_other"

    # Ensure all required columns exist before computing
    if all(col in pivot_extended.columns for col in [shots_col, winners_col, induced_col, unforced_col]):
        pivot_extended[other_col] = (
            pivot_extended[shots_col]
            - pivot_extended[winners_col]
            - pivot_extended[induced_col]
            - pivot_extended[unforced_col]
        )


print(pivot_extended[0:10])


shot_cols = [col for col in shots_pivot.columns if col not in ["player"]]

for x in shot_cols:
    if_col = x + "_induced_forced"
    shots_col = x + "_shots"
    unf_col = x + "_unforced"
    win_col = x + "_winners"


outcome_cols = ['winners', 'induced_forced', 'unforced']

shots_percent = shots_pivot.copy()
shots_percent[shot_cols] = shots_percent[shot_cols].div(shots_percent["Total"], axis=0) * 100

# print(shots_percent)

sum_cols = ["shots", "pt_ending", "winners", "induced_forced", "unforced"]

summed_df = data.groupby(["player", "row"], as_index=False)[sum_cols].sum()

print(summed_df[0:50])

summed_df['winner_pct'] = (summed_df['winners'] / summed_df['shots']) * 100
summed_df['induced_forced_pct'] = (summed_df['induced_forced'] / summed_df['shots']) * 100
summed_df['unforced_pct'] = (summed_df['unforced'] / summed_df['shots']) * 100

summed_df['non_pt_ending'] = summed_df['shots'] - summed_df['pt_ending']
summed_df['non_pt_ending_pct'] = (summed_df['non_pt_ending'] / summed_df['shots']) * 100

summed_df[['winner_pct', 'induced_forced_pct', 'unforced_pct', 'non_pt_ending_pct']] = \
    summed_df[['winner_pct', 'induced_forced_pct', 'unforced_pct', 'non_pt_ending_pct']].fillna(0)

total_shots_df = summed_df[summed_df["row"] == "Total"][["player", "shots"]].rename(columns={"shots": "total_player_shots"})

summed_df = pd.merge(summed_df, total_shots_df, on="player", how="left")

summed_df["shot_usage_pct"] = (summed_df["shots"] / summed_df["total_player_shots"]) * 100

long_df = summed_df[[
    "player", "row", "shot_usage_pct", "winner_pct", "induced_forced_pct", "unforced_pct", "non_pt_ending_pct"
]].rename(columns={"row": "shot_type"})

print(long_df.head(50))

print(long_df.shape)

# Define the mapping from original shot_type to descriptive labels
shot_type_map = {
    'Fside': 'forehand_side',
    'Bside': 'backhand_side',
    'F': 'forehand_groundstroke',
    'B': 'backhand_groundstroke',
    'R': 'forehand_slice',
    'S': 'backhand_slice',
    'V': 'forehand_volley',
    'Z': 'backhand_volley',
    'O': 'overhead_smash',
    'P': 'backhand_overhead_smash',
    'U': 'forehand_drop_shot',
    'Y': 'backhand_drop_shot',
    'L': 'forehand_lob',
    'M': 'backhand_lob',
    'H': 'forehand_half_volley',
    'I': 'backhand_half_volley',
    'J': 'forehand_swinging_volley',
    'K': 'backhand_swinging_volley'
}

# Filter and rename shot types
filtered_df = long_df[long_df['shot_type'].isin(shot_type_map.keys())].copy()
filtered_df['shot_type'] = filtered_df['shot_type'].map(shot_type_map)

# Pivot to wide format
reshaped_df = filtered_df.pivot(index='player', columns='shot_type', values=[
    'shot_usage_pct', 'winner_pct', 'induced_forced_pct', 'unforced_pct'
])

# Flatten MultiIndex columns
reshaped_df.columns = [f"{shot}_{metric}" for metric, shot in reshaped_df.columns]

# Reset index
reshaped_df = reshaped_df.reset_index()

# Define custom shot type order
shot_order = [
    'forehand_side', 'backhand_side',
    'forehand_groundstroke', 'backhand_groundstroke',
    'forehand_slice', 'backhand_slice',
    'forehand_volley', 'backhand_volley',
    'overhead_smash', 'backhand_overhead_smash',
    'forehand_drop_shot', 'backhand_drop_shot',
    'forehand_lob', 'backhand_lob',
    'forehand_half_volley', 'backhand_half_volley',
    'forehand_swinging_volley', 'backhand_swinging_volley'
]

# Reorder columns by shot type prefix
ordered_cols = ['player']
for shot in shot_order:
    ordered_cols += [col for col in reshaped_df.columns if col.startswith(shot)]

# Final reordered DataFrame
reshaped_df = reshaped_df[ordered_cols]

# Display result
print(reshaped_df.head(50))

csv_data = reshaped_df.to_csv('player_shot_types.csv')
