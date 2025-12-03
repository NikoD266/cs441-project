import pandas as pd
import glob
from collections import defaultdict
import os

Baseline_Rating = 1500
K_factor = 32  # the k factor is the multiplier that determine how much a win/loss affects rating
Surface_Weight = 0.5  # weight of surface Elo when predicting
output = "Elo_Output"
os.makedirs(output, exist_ok=True)  # makes our folder that will contain our output files


def load_all_matches(pattern="tennis_atp/atp_matches_2*.csv"):
    files = sorted(glob.glob(pattern))  # gathers the files according to the pattern and then sorts them
    dfs = []
    for each in files:
        df = pd.read_csv(each, low_memory=False)
        df["source_file"] = os.path.basename(each)  # Makes a column for the name of each file
        dfs.append(df)
    all_matches = pd.concat(dfs, ignore_index=True)
    return all_matches


def parse_date_from_csv(x):  # tourney_date in dataset is in the form of an integer
    if pd.isnull(x):
        return pd.NaT
    s = str(int(x))
    return pd.to_datetime(s, format="%Y%m%d")


def expected_score(rating1, rating2):  # Traditional Elo model
    return 1.0 / (1.0 + 10 ** ((rating2 - rating1) / 400.0))


def update_elo(rating, opp_rating, score, k):  # Score is just binary 1.0 or 0.0
    return rating + k * (score - expected_score(rating, opp_rating))


def prepare_matches(df):
    df["tourney_date_parsed"] = df["tourney_date"].apply(parse_date_from_csv)
    df = df.sort_values("tourney_date_parsed").reset_index(drop=True)
    return df


def run_elo(df, k=K_factor, baseline_rating=Baseline_Rating):
    ratings_overall = defaultdict(lambda: baseline_rating)
    ratings_surface = defaultdict(lambda: baseline_rating)  # keyed by (player_id, surface)
    player_last_match = {}  # I am creating this, so we only have current players for our ratings.

    # store a record for every single match.
    history = []
    for idx, row in df.iterrows():
        date = row.get("tourney_date_parsed", pd.NaT)
        surface = row.get("surface")
        winner_name = row["winner_name"]
        loser_name = row["loser_name"]

        # update last match date for both players
        player_last_match[winner_name] = date
        player_last_match[loser_name] = date

        winner_rating_overall = ratings_overall[winner_name]
        loser_rating_overall = ratings_overall[loser_name]
        winner_rating_surface = ratings_surface[(winner_name, surface)]
        loser_rating_surface = ratings_surface[(loser_name, surface)]
        # prediction using surface ratings that could be used for evaluation later
        blended_rating_winner = Surface_Weight * winner_rating_surface + (1 - Surface_Weight) * winner_rating_overall
        blended_rating_loser = Surface_Weight * loser_rating_surface + (1 - Surface_Weight) * loser_rating_overall
        # compute expected probability
        exp_w = expected_score(blended_rating_winner, blended_rating_loser)

        # Update ratings: winner gets 1, loser 0 and then update the overall ratings
        new_overall_rating_winner = update_elo(winner_rating_overall, loser_rating_overall, 1.0, k)
        new_overall_rating_loser = update_elo(loser_rating_overall, winner_rating_overall, 0.0, k)
        ratings_overall[winner_name] = new_overall_rating_winner
        ratings_overall[loser_name] = new_overall_rating_loser

        # Update surface ratings
        new_surface_rating_winner = update_elo(winner_rating_surface, loser_rating_surface, 1.0, k)
        new_surface_rating_loser = update_elo(loser_rating_surface, winner_rating_surface, 0.0, k)
        ratings_surface[(winner_name, surface)] = new_surface_rating_winner
        ratings_surface[(loser_name, surface)] = new_surface_rating_loser

        # record history row
        history.append({
            "date": date,
            "surface": surface,
            "winner_id": row.get("winner_id"),
            "loser_id": row.get("loser_id"),
            "winner_name": winner_name,
            "loser_name": loser_name,
            "predicted_win_prob": exp_w,
            "actual_winner": 1,
            "r_w_global_before": winner_rating_overall,
            "r_l_global_before": loser_rating_overall,
            "r_w_global_after": ratings_overall[winner_name],
            "r_l_global_after": ratings_overall[loser_name],
            "r_w_surface_before": winner_rating_surface,
            "r_l_surface_before": loser_rating_surface
        })

    history_df = pd.DataFrame(history)
    return ratings_overall, ratings_surface, history_df, player_last_match


def save_outputs(ratings_global, ratings_surface, history_df, player_last_match, cutoff_year=2024):
    global_df = pd.DataFrame([
        {"player_name": pid, "rating": r, "last_match": player_last_match.get(pid, pd.NaT)}
        for pid, r in ratings_global.items()
    ]).sort_values("rating", ascending=False)
    global_df["last_match"] = pd.to_datetime(global_df["last_match"], errors="coerce")
    global_df.to_csv(os.path.join(output, "final_global_ratings.csv"), index=False)

    # Filter to current players
    mask_current = global_df["last_match"].dt.year == int(cutoff_year)
    current_df = global_df[mask_current].sort_values("rating", ascending=False)
    current_df.to_csv(os.path.join(output, f"final_global_ratings_current_players_{cutoff_year}.csv"), index=False)

    # surface ratings (all)
    surf_rows = []
    for (pid, surf), r in ratings_surface.items():
        surf_rows.append({
            "player_name": pid,
            "surface": surf,
            "rating": r,
            "last_match": player_last_match.get(pid, pd.NaT)
        })
    surf_df = pd.DataFrame(surf_rows).sort_values(["surface", "rating"], ascending=[True, False])
    surf_df["last_match"] = pd.to_datetime(surf_df["last_match"], errors="coerce")
    surf_df.to_csv(os.path.join(output, "final_surface_ratings.csv"), index=False)

    surf_current_df = surf_df[surf_df["last_match"].dt.year == int(cutoff_year)]
    surf_current_df.to_csv(os.path.join(output, f"final_surface_ratings_current_players_{cutoff_year}.csv"), index=False)

    # Compute Brier per-row
    # make a copy
    hist = history_df.copy()
    # per-row Brier score
    hist["brier"] = (hist["predicted_win_prob"].astype(float) - hist["actual_winner"].astype(float)) ** 2
    # extract year for grouping
    hist["year"] = hist["date"].dt.year
    # Save the  history
    hist.to_csv(os.path.join(output, "elo_history_with_brier.csv"), index=False)

    # Yearly Brier summary
    summary = (
        hist.groupby("year", as_index=True).agg(matches=("brier", "count"), mean_brier=("brier", "mean")).sort_index()
        .reset_index()
    )
    summary.to_csv(os.path.join(output, "brier_summary_by_year.csv"), index=False)
    # Overall Brier summary
    overall = {
        "total_matches": int(hist["brier"].count()),
        "overall_mean_brier": float(hist["brier"].mean()) if hist["brier"].count() > 0 else None
    }
    pd.DataFrame([overall]).to_csv(os.path.join(output, "overall_brier_summary.csv"), index=False)
    print(f"Saved all csv outputs into {output}")

def add_elo_to_matches(original_df, history_df):
    # Keep only relevant Elo columns and identifying info
    elo_cols = [
        "winner_name", "loser_name", "date",
        "r_w_global_before", "r_l_global_before",
        "r_w_surface_before", "r_l_surface_before"
    ]
    elo_info = history_df[elo_cols].copy()
    
    # Merge on winner, loser, and date
    merged_df = original_df.merge(
        elo_info,
        left_on=["winner_name", "loser_name", "tourney_date_parsed"],
        right_on=["winner_name", "loser_name", "date"],
        how="left"
    )
    merged_df.drop(columns=["date"], inplace=True)  # drop extra date column from merge
    return merged_df

def save_matches_with_elo(df):
    for fname in df["source_file"].unique():
        sub_df = df[df["source_file"] == fname].copy()
        out_path = os.path.join(output, f"elo_{fname}")
        sub_df.to_csv(out_path, index=False)
        print(f"Saved Elo-enhanced file: {out_path}")

'''
def main():
    print("Loading matches...")
    all_matches = load_all_matches()
    print(f"Loaded {len(all_matches)} rows from CSVs.")
    matches = prepare_matches(all_matches)
    print(f"{len(matches)} matches after preparation.")
    print("Running Elo updates...")
    ratings_global, ratings_surface, history_df, player_last_match = run_elo(matches)
    save_outputs(ratings_global, ratings_surface, history_df, player_last_match, cutoff_year=2024)
    print("Outputs are updated in CSVs")
'''
def main():
    print("Loading matches...")
    all_matches = load_all_matches()
    print(f"Loaded {len(all_matches)} rows from CSVs.")
    
    matches = prepare_matches(all_matches)
    print(f"{len(matches)} matches after preparation.")
    
    print("Running Elo updates...")
    ratings_global, ratings_surface, history_df, player_last_match = run_elo(matches)
    
    print("Adding Elo columns to original matches...")
    matches_with_elo = add_elo_to_matches(matches, history_df)
    
    print("Saving Elo-enhanced matches...")
    save_matches_with_elo(matches_with_elo)
    
    print("Saving other Elo outputs...")
    save_outputs(ratings_global, ratings_surface, history_df, player_last_match, cutoff_year=2024)
    print("All outputs updated.")


if __name__ == "__main__":
    main()
