import os

import joblib

from config import (
    RATINGS_PATH,
    MOVIES_PATH,
    LINKS_PATH,
    MIN_USER_RATINGS,
    MIN_MOVIE_RATINGS,
    N_COMPONENTS,
    RANDOM_STATE,
    POPULARITY_MIN_RATINGS,
    TOP_N,
)
from data_loader import load_data, filter_data, split_train_test
from models import SVDRecommender, ContentBasedRecommender, popularity_recommend


def train_and_save(
    out_path: str = "artifacts/recommender.joblib",
    top_n_popular: int = 50,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("Loading data...")
    ratings, movies = load_data(RATINGS_PATH, MOVIES_PATH, LINKS_PATH)

    print("Filtering data...")
    ratings = filter_data(
        ratings,
        min_user_ratings=MIN_USER_RATINGS,
        min_movie_ratings=MIN_MOVIE_RATINGS,
    )
    if len(ratings) == 0:
        raise ValueError("Không còn dữ liệu sau khi lọc. Hãy giảm ngưỡng trong config.py")

    print("Splitting train/test...")
    train_df, _ = split_train_test(ratings)

    print("Training SVD model...")
    model = SVDRecommender(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    explained_var = model.fit(train_df, movies)

    print("Precomputing popular movies...")
    popular_df = popularity_recommend(
        train_df,
        movies,
        top_n=top_n_popular,
        min_ratings=POPULARITY_MIN_RATINGS,
    )

    print("Training Content-Based model...")
    cb_model = ContentBasedRecommender()
    cb_model.fit(train_df, movies)

    artifacts = {
        "model": model,
        "cb_model": cb_model,
        "popular_df": popular_df,
        "movies_df": movies,
        "meta": {
            "explained_variance_sum": float(explained_var),
            "top_n_default": int(TOP_N),
            "popularity_min_ratings_default": int(POPULARITY_MIN_RATINGS),
            "min_user_ratings": int(MIN_USER_RATINGS),
            "min_movie_ratings": int(MIN_MOVIE_RATINGS),
            "n_components": int(model.n_components),
            "random_state": int(model.random_state),
        },
    }

    print(f"Saving artifacts to {out_path} ...")
    joblib.dump(artifacts, out_path, compress=3)
    print("Done.")


if __name__ == "__main__":
    train_and_save()

