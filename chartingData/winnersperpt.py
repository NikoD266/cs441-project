import pandas as pd

df = pd.read_csv("charting-m-stats-ShotTypes.csv")
df = df[['player', 'row', 'pt_ending', 'winners', 'induced_forced', 'unforced']]

df_totals = df[df['row'] == 'Total'].copy()

career_totals = df_totals.groupby('player', as_index=False).sum(numeric_only=True)
career_totals['winners_per_point'] = career_totals['winners'] / career_totals['pt_ending']
career_totals['forced_per_point'] = career_totals['induced_forced'] / career_totals['pt_ending']
career_totals['unforced_per_point'] = career_totals['unforced'] / career_totals['pt_ending']

output = career_totals[['player', 'winners_per_point', 'forced_per_point', 'unforced_per_point']]
print(output)
output.to_csv("career_point_stats.csv", index=False)


