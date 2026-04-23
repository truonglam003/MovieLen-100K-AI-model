import numpy as np


def evaluate_popularity_model(
    train_df,
    test_df,
    movies_df,
    popularity_fn,
    k=10,
    min_ratings=50,
    max_users=500
):
    top_popular_ids = popularity_fn(
        train_df,
        movies_df,
        top_n=k,
        min_ratings=min_ratings
    )["movieId"].tolist()

    precisions = []
    recalls = []

    grouped_test = list(test_df.groupby("userId"))[:max_users]

    for _, user_test_rows in grouped_test:
        positive_items = set(user_test_rows["movieId"].tolist())

        hits = sum(1 for movie_id in top_popular_ids if movie_id in positive_items)

        precisions.append(hits / k)
        recalls.append(hits / len(positive_items))

    if len(precisions) == 0:
        return {
            "evaluated_users": 0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0
        }

    return {
        "evaluated_users": len(precisions),
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls))
    }


def evaluate_svd_model(
    model,
    test_df,
    k=10,
    negative_sample_size=30,
    random_state=42,
    max_users=500
):
    precisions = []
    recalls = []

    rng = np.random.default_rng(random_state)
    grouped_test = list(test_df.groupby("userId"))[:max_users]

    evaluated_users = 0

    for user_id, user_test_rows in grouped_test:
        if user_id not in model.user_id_to_index:
            continue

        positive_items = set(user_test_rows["movieId"].tolist())
        seen_items = model.user_seen_movies.get(user_id, set())

        candidate_pool = list(model.all_train_movie_ids - seen_items - positive_items)
        if len(candidate_pool) == 0:
            continue

        sample_size = min(negative_sample_size, len(candidate_pool))
        negative_items = set(
            rng.choice(candidate_pool, size=sample_size, replace=False).tolist()
        )

        candidate_items = list(positive_items | negative_items)

        user_idx = model.user_id_to_index[user_id]
        user_vector = model.user_factors[user_idx]
        base_mean = float(model.user_mean_rating[user_idx])

        scored_items = []

        for movie_id in candidate_items:
            if movie_id not in model.movie_id_to_index:
                continue

            movie_idx = model.movie_id_to_index[movie_id]
            score = float(user_vector @ model.item_factors[movie_idx] + base_mean)
            scored_items.append((movie_id, score))

        if len(scored_items) == 0:
            continue

        scored_items.sort(key=lambda x: x[1], reverse=True)
        top_k_items = [movie_id for movie_id, _ in scored_items[:k]]

        hits = sum(1 for movie_id in top_k_items if movie_id in positive_items)

        precisions.append(hits / k)
        recalls.append(hits / len(positive_items))
        evaluated_users += 1

    if evaluated_users == 0:
        return {
            "evaluated_users": 0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0
        }

    return {
        "evaluated_users": evaluated_users,
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls))
    }