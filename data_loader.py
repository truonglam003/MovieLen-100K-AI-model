import pandas as pd


def load_data(ratings_path, movies_path, links_path=None):
    ratings = pd.read_csv(
        ratings_path,
        usecols=["userId", "movieId", "rating"],
        dtype={
            "userId": "int32",
            "movieId": "int32",
            "rating": "float32"
        }
    )

    movies = pd.read_csv(
        movies_path,
        usecols=["movieId", "title", "genres"],
        dtype={
            "movieId": "int32",
            "title": "string",
            "genres": "string"
        }
    )

    if links_path is not None:
        links = pd.read_csv(
            links_path,
            usecols=["movieId", "tmdbId"],
            dtype={
                "movieId": "int32",
                "tmdbId": "Int64"  # Int64 supports NaNs
            }
        )
        movies = movies.merge(links, on="movieId", how="left")

    return ratings, movies


def filter_data(ratings, min_user_ratings=20, min_movie_ratings=20):
    user_counts = ratings["userId"].value_counts()
    movie_counts = ratings["movieId"].value_counts()

    valid_users = user_counts[user_counts >= min_user_ratings].index
    valid_movies = movie_counts[movie_counts >= min_movie_ratings].index

    filtered = ratings[
        ratings["userId"].isin(valid_users) &
        ratings["movieId"].isin(valid_movies)
    ].copy()

    return filtered


def split_train_test(ratings):
    ratings = ratings.reset_index(drop=True)

    test_idx = ratings.groupby("userId").tail(1).index
    test_df = ratings.loc[test_idx].copy()
    train_df = ratings.drop(test_idx).copy()

    train_user_counts = train_df["userId"].value_counts()
    valid_train_users = train_user_counts[train_user_counts >= 1].index

    train_df = train_df[train_df["userId"].isin(valid_train_users)].copy()
    test_df = test_df[test_df["userId"].isin(valid_train_users)].copy()

    return train_df, test_df