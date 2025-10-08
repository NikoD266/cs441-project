import pandas as pd


atp_matches_list = [
    "atp_matches_2000",
]

df = pd.read_csv("./tennis_atp/atp_matches_2000.csv")

print(df.head())
print(df.info())
print(df.shape) 




total_rows = 0
combined_df_list = []
for year in range(2000,2025):
    csv_name = f"./tennis_atp/atp_matches_{year}.csv"
    df = pd.read_csv(csv_name)
    total_rows += df.shape[0]
    
    missing_value_counter = df['tourney_date'].isnull().sum() + df['tourney_level'].isnull().sum() + df['winner_id'].isnull().sum() + df['winner_name'].isnull().sum() + \
        df['winner_hand'].isnull().sum() + df['loser_id'].isnull().sum() + df['loser_name'].isnull().sum() + df['loser_hand'].isnull().sum()
    if missing_value_counter != 0:
        print("null count:", missing_value_counter)
        print("number of nulls in tourney_date column:",df['tourney_date'].isnull().sum())
        print("number of nulls in tourney_level column:",df['tourney_level'].isnull().sum())
        print("number of nulls in winner_id column:",df['winner_id'].isnull().sum())
        print("number of nulls in winner_name column:",df['winner_name'].isnull().sum())
        print("number of nulls in winner_hand column:",df['winner_hand'].isnull().sum())
        print("number of nulls in loser_id column:",df['loser_id'].isnull().sum())
        print("number of nulls in loser_name column:",df['loser_name'].isnull().sum())
        print("number of nulls in loser_hand column:",df['loser_hand'].isnull().sum())
    else:
        print("No null values in:",csv_name)
    duplicate_matches = df.duplicated(subset=['tourney_date', 'match_num','tourney_id']).sum()
    if duplicate_matches != 0:
        print("Duplicate matches--------------------:", duplicate_matches)

    invalid_matches = (df['winner_id'] == df['loser_id']).sum()
    if invalid_matches != 0:
        print("Matches where winner equals loser:", invalid_matches)
    dupes = df[df.duplicated(keep=False)]
    if len(dupes) != 0:
        print("Number of full-row duplicates:", len(dupes))
    duplicates_ignore_case = df['winner_name'].str.lower().duplicated().sum()
    
    if duplicates_ignore_case != 0:
        print("Duplicates ignoring case:", duplicates_ignore_case)
    #print(df['winner_hand'].value_counts(dropna=False))
    #print(df['loser_hand'].value_counts(dropna=False))
    #print(df['tourney_level'].value_counts(dropna=False))
    combined_df_list.append(df)
all_data = pd.concat(combined_df_list, ignore_index = True)
counts = all_data['winner_hand'].value_counts(dropna=False)
percentages = counts / counts.sum() * 100
print("Total number of rows from atp_matches 2000->2024:", total_rows)    
print("percentage:", percentages)

'''
df = pd.read_csv("./chartingData/charting-m-stats-Overview.csv")
print(df.head())
overview_columns= ["player","aces", "dfs", "first_in", "first_won", "second_in", "second_won", "return_pts_won", "winners", "unforced"]

for column in overview_columns:
    print("number of nulls in column" ,column ,":", df[column].isnull().sum())
    print(column, df[column].min(), df[column].max())
dupes = df[df.duplicated(keep=False)]

print("Number of full-row duplicates:", len(dupes))

df = pd.read_csv("./chartingData/charting-m-stats-ShotDirOutcomes.csv")
print(df.head())
shotDirOutcomes_columns = ["match_id","player","pt_ending","winners","induced_forced","unforced","shots_in_pts_won","shots_in_pts_lost"]

for column in shotDirOutcomes_columns:
    print("number of nulls in column" ,column ,":", df[column].isnull().sum())
    print(column, "min:", df[column].min(), "max:", df[column].max())

dupes = df[df.duplicated(keep=False)]

print("Number of full-row duplicates:", len(dupes))


shot_direction_columns = [
    "player",
    "crosscourt",
    "down_middle",
    "down_the_line",
    "inside_out",
    "inside_in"
]
df = pd.read_csv("./chartingData/charting-m-stats-ShotDirection.csv")
print(df.head())
for column in shot_direction_columns:
    print("number of nulls in column" ,column ,":", df[column].isnull().sum())
    print(column, "min:", df[column].min(), "max:", df[column].max())
dupes = df[df.duplicated(keep=False)]

print("Number of full-row duplicates:", len(dupes))
    

df = pd.read_csv("./chartingData/charting-m-stats-ShotTypes.csv")
print(df.head())
shot_type_columns = [
    "player",
    "winners",
    "induced_forced",
    "unforced",
    "serve_return",
    "shots_in_pts_won",
    "shots_in_pts_lost"
]
for column in shot_type_columns:
    print("number of nulls in column" ,column ,":", df[column].isnull().sum())
    print(column, "min:", df[column].min(), "max:", df[column].max())
dupes = df[df.duplicated(keep=False)]

print("Number of full-row duplicates:", len(dupes))'''