import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



if __name__ == '__main__':
    overview_df = pd.read_csv("chartingData/charting-m-stats-Overview.csv")
    overview_20xx_df = overview_df[(overview_df["match_id"].str.contains('^20', case=False, regex=True)) & (overview_df['set'] == "Total")]

    print(overview_20xx_df)

    player_data_df = overview_20xx_df[['player', 'aces', 'dfs', 'winners']].groupby('player').mean().reset_index()
    player_data_df = player_data_df.drop('aces', axis = 1)


    percents_df_full = overview_20xx_df[['player', 'serve_pts', 'first_in', 'first_won', 'second_in', 'second_won', 'return_pts', 'return_pts_won', 'winners', 'unforced', 'dfs']].groupby('player').sum().reset_index()
    percents_df_full['first_in_percent'] = percents_df_full['first_in'] / percents_df_full['serve_pts']
    percents_df_full['first_won_percent'] = percents_df_full['first_won'] / percents_df_full['first_in']
    percents_df_full['second_won_percent'] = percents_df_full['second_won'] / percents_df_full['second_in']
    percents_df_full['return_points_won_percent'] = percents_df_full['return_pts_won'] / percents_df_full['return_pts']
    percents_df_full['winners_percent'] = percents_df_full['winners'] / (percents_df_full['serve_pts'] + percents_df_full['return_pts'])
    percents_df_full['unforced_percent'] = percents_df_full['unforced'] / (percents_df_full['serve_pts'] + percents_df_full['return_pts'])
    percents_df_full['df_percent'] = percents_df_full['dfs'] / percents_df_full['serve_pts']

    percents_df = percents_df_full[['player', 'first_in_percent', 'first_won_percent', 'second_won_percent', 'return_points_won_percent', 'winners_percent', 'unforced_percent', 'df_percent']]

    print(overview_20xx_df)

    # plt.hist(player_data_df['aces'].tolist())
    # plt.title('Player Avg Aces Per Match')
    # plt.show()
    #
    # plt.hist(player_data_df['dfs'].tolist())
    # plt.title('Player Avg Double Faults Per Match')
    # plt.show()
    #
    # plt.hist(player_data_df['winners'].tolist())
    # plt.title('Player Avg \'Winners\' Per Match')
    # plt.show()
    #
    # plt.hist(percents_df['first_in_percent'].tolist())
    # plt.title('Percent of First Serve Being In')
    # plt.show()
    #
    # plt.hist(percents_df['first_won_percent'].tolist())
    # plt.title('Percent of First Serve Points Won')
    # plt.show()
    #
    # plt.hist(percents_df['second_won_percent'].tolist())
    # plt.title('Percent of Second Serve Points Won')
    # plt.show()
    #
    # plt.hist(percents_df['return_points_won_percent'].tolist())
    # plt.title('Percent of Return Points Won')
    # plt.show()
    #
    # plt.hist(percents_df['winners_percent'].tolist())
    # plt.title('Percent of Points Ending in A \'Winner\' Shot')
    # plt.show()

    # plt.hist(percents_df['unforced_percent'].tolist())
    # plt.title('Percent of Points Ending in A Unforced Error')
    # plt.show()

    serveBasics = pd.read_csv("chartingData/charting-m-stats-ServeBasics.csv")
    serveBasics = serveBasics[serveBasics['row'] == "Total"]
    serveBasics['aces_and_unret'] = serveBasics['aces'] + serveBasics['unret']
    player_unret = serveBasics[['player', 'unret']].groupby('player').mean().reset_index()
    player_aces_and_unret = serveBasics[['player', 'pts', 'aces_and_unret']].groupby('player').sum().reset_index()
    player_aces_and_unret['aces_and_unret_percentage'] = player_aces_and_unret['aces_and_unret'] / player_aces_and_unret['pts']
    print(player_unret)

    # player_full_df = pd.merge(player_data_df, percents_df, on="player", how='inner')
    player_full_df = pd.merge(percents_df, player_aces_and_unret[['player', 'aces_and_unret_percentage']], on='player', how='inner')

    print(player_full_df)
    # points_2020s_df = pd.read_csv("chartingData/charting-m-points-2020s.csv")
    # latest_match_df = overview_20xx_df['match_id'][0]
    # pts_from_latest_match = points_2020s_df[points_2020s_df['match_id'] == latest_match_df]
    # p1_serves = len(pts_from_latest_match[pts_from_latest_match['Svr'] == 1])
    # p2_serves = len(pts_from_latest_match[pts_from_latest_match['Svr'] == 2])


    # print(points_2020s_df)





















