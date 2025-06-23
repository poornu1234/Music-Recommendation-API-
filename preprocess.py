
import pandas as pd

def load_and_clean(path="data/ratings.csv"):
    df = pd.read_csv(path)

    # Group by Username + Track → count play frequency
    df_counts = df.groupby(["Username", "Track"]).size().reset_index(name="play_count")

    # Convert play_count to rating (1-5 scale using quantiles)
    df_counts["rating"] = pd.qcut(df_counts["play_count"], 5, labels=[1, 2, 3, 4, 5]).astype(int)

    return df_counts
