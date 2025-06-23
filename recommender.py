
from surprise import Dataset, Reader, SVD

def train_model(df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["Username", "Track", "rating"]], reader)

    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)

    return algo

def get_recommendations(algo, df, user_id, top_n=5):
    all_tracks = df["Track"].unique()
    rated = df[df["Username"] == user_id]["Track"].tolist()

    if not rated:
        return [{"error": "User ID not found or has no data"}]

    unrated_tracks = set(all_tracks) - set(rated)

    predictions = []
    for track in unrated_tracks:
        pred = algo.predict(user_id, track)
        predictions.append((track, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return [{"track": t, "predicted_rating": round(r, 2)} for t, r in predictions[:top_n]]
