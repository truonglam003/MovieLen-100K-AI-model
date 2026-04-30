import pandas as pd

def load_data(engine):
    print("Fetching ratings from MySQL...")
    ratings_query = "SELECT user_id AS userId, movie_id AS movieId, rating FROM ratings"
    ratings = pd.read_sql(ratings_query, engine)
    
    ratings = ratings.astype({
        "userId": "int32",
        "movieId": "int32",
        "rating": "float32"
    })

    print("Fetching movies from MySQL...")
    movies_query = "SELECT id AS movieId, title, genres, tmdb_id AS tmdbId FROM movies"
    movies = pd.read_sql(movies_query, engine)
    
    movies = movies.astype({
        "movieId": "int32",
        "title": "string",
        "genres": "string",
        "tmdbId": "Int64"
    })

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