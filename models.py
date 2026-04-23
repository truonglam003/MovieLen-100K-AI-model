import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def popularity_recommend(train_data, movies_df, top_n=10, min_ratings=50):
    movie_stats = train_data.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index()

    movie_stats = movie_stats[movie_stats["rating_count"] >= min_ratings].copy()

    if len(movie_stats) == 0:
        movie_stats = train_data.groupby("movieId").agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count")
        ).reset_index()

    C = movie_stats["avg_rating"].mean()
    m = movie_stats["rating_count"].quantile(0.60)

    movie_stats["score"] = (
        (movie_stats["rating_count"] / (movie_stats["rating_count"] + m)) * movie_stats["avg_rating"]
        + (m / (movie_stats["rating_count"] + m)) * C
    )

    top_movies = movie_stats.sort_values(
        by=["score", "rating_count"],
        ascending=[False, False]
    ).head(top_n)

    top_movies = top_movies.merge(movies_df, on="movieId", how="left")

    columns_to_return = ["movieId", "title", "genres", "avg_rating", "rating_count", "score"]
    if "tmdbId" in top_movies.columns:
        columns_to_return.append("tmdbId")
    
    return top_movies[columns_to_return]


class SVDRecommender:
    def __init__(self, n_components=30, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

        self.user_id_to_index = None
        self.movie_id_to_index = None
        self.index_to_movie_id = None

        self.user_mean_rating = None  # np.ndarray, shape (n_users,)
        self.user_factors = None
        self.item_factors = None
        self.user_seen_movies = None
        self.all_train_movie_ids = None
        self.movies_df = None

    def fit(self, train_df, movies_df):
        self.movies_df = movies_df.copy()

        unique_users = np.sort(train_df["userId"].unique())
        unique_movies = np.sort(train_df["movieId"].unique())

        self.user_id_to_index = {uid: idx for idx, uid in enumerate(unique_users)}
        self.movie_id_to_index = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.index_to_movie_id = {idx: mid for mid, idx in self.movie_id_to_index.items()}

        train_df = train_df.copy()
        train_df["user_idx"] = train_df["userId"].map(self.user_id_to_index).astype("int32")
        train_df["movie_idx"] = train_df["movieId"].map(self.movie_id_to_index).astype("int32")

        n_users = len(unique_users)
        n_movies = len(unique_movies)

        # mean rating theo user index (vector hoá để recommend nhanh hơn)
        user_means_series = train_df.groupby("user_idx")["rating"].mean().astype("float32")
        self.user_mean_rating = np.zeros(n_users, dtype=np.float32)
        self.user_mean_rating[user_means_series.index.to_numpy(dtype=np.int32)] = user_means_series.to_numpy()

        user_idx_arr = train_df["user_idx"].to_numpy(dtype=np.int32, copy=False)
        centered_ratings = train_df["rating"].to_numpy(dtype=np.float32, copy=False) - self.user_mean_rating[user_idx_arr]
        centered_ratings = centered_ratings.astype("float32")

        user_item_sparse = csr_matrix(
            (centered_ratings, (user_idx_arr, train_df["movie_idx"].to_numpy(dtype=np.int32, copy=False))),
            shape=(n_users, n_movies),
            dtype=np.float32
        )

        max_components = min(n_users - 1, n_movies - 1, self.n_components)
        if max_components < 2:
            raise ValueError("Dữ liệu quá nhỏ để train SVD.")

        svd = TruncatedSVD(
            n_components=max_components,
            algorithm="randomized",
            n_iter=7,
            random_state=self.random_state
        )

        self.user_factors = svd.fit_transform(user_item_sparse).astype(np.float32)
        self.item_factors = svd.components_.T.astype(np.float32)

        user_ids = train_df["userId"].to_numpy()
        movie_ids = train_df["movieId"].to_numpy()
        max_user_id = user_ids.max()
        
        self.seen_sparse = csr_matrix(
            (np.ones(len(user_ids), dtype=bool), (user_ids, movie_ids)),
            shape=(max_user_id + 1, self.movies_df['movieId'].max() + 1)
        )

        self.all_train_movie_ids = set(train_df["movieId"].unique())

        return svd.explained_variance_ratio_.sum()

    def recommend(self, user_id, top_n=10):
        if user_id not in self.user_id_to_index:
            return None

        user_idx = self.user_id_to_index[user_id]

        scores = self.user_factors[user_idx] @ self.item_factors.T
        scores = scores + float(self.user_mean_rating[user_idx])

        if hasattr(self, 'seen_sparse') and user_id < self.seen_sparse.shape[0]:
            s_start = self.seen_sparse.indptr[user_id]
            s_end = self.seen_sparse.indptr[user_id+1]
            seen = self.seen_sparse.indices[s_start:s_end]
        else:
            seen = []
            
        seen_indices = [self.movie_id_to_index[mid] for mid in seen if mid in self.movie_id_to_index]

        scores = scores.astype(np.float32)
        scores[seen_indices] = -np.inf

        candidate_count = min(top_n * 5, len(scores))
        top_idx_unsorted = np.argpartition(scores, -candidate_count)[-candidate_count:]
        top_idx_sorted = top_idx_unsorted[np.argsort(scores[top_idx_unsorted])[::-1]]

        top_movie_indices = top_idx_sorted[:top_n]
        top_movie_ids = [self.index_to_movie_id[idx] for idx in top_movie_indices]
        top_scores = [float(scores[idx]) for idx in top_movie_indices]

        recs = pd.DataFrame({
            "movieId": top_movie_ids,
            "pred_score": top_scores
        }).merge(self.movies_df, on="movieId", how="left")

        columns_to_return = ["movieId", "title", "genres", "pred_score"]
        if "tmdbId" in recs.columns:
            columns_to_return.append("tmdbId")
            
        return recs[columns_to_return]


class ContentBasedRecommender:
    def __init__(self):
        self.movies_df = None
        self.tfidf_matrix = None
        self.movie_id_to_index = None
        self.index_to_movie_id = None
        self.user_liked_movies = None
        self.user_seen_movies = None

    def fit(self, train_df, movies_df):
        self.movies_df = movies_df.copy()
        
        # Tiền xử lý genres
        self.movies_df['genres_processed'] = self.movies_df['genres'].fillna('').str.replace('|', ' ', regex=False)
        
        vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = vectorizer.fit_transform(self.movies_df['genres_processed'])
        
        self.movie_id_to_index = {mid: idx for idx, mid in enumerate(self.movies_df['movieId'])}
        self.index_to_movie_id = {idx: mid for mid, idx in self.movie_id_to_index.items()}
        
        # Xây dựng ma trận thưa cho seen và liked để tối ưu hóa memory và joblib dump
        user_ids = train_df["userId"].to_numpy()
        movie_ids = train_df["movieId"].to_numpy()
        ratings = train_df["rating"].to_numpy()
        
        max_user_id = user_ids.max()
        
        self.seen_sparse = csr_matrix(
            (np.ones(len(user_ids), dtype=bool), (user_ids, movie_ids)),
            shape=(max_user_id + 1, self.movies_df['movieId'].max() + 1)
        )
        
        liked_mask = ratings >= 3.5
        self.liked_sparse = csr_matrix(
            (np.ones(liked_mask.sum(), dtype=bool), (user_ids[liked_mask], movie_ids[liked_mask])),
            shape=(max_user_id + 1, self.movies_df['movieId'].max() + 1)
        )
        
    def recommend(self, user_id, top_n=10):
        if not hasattr(self, 'liked_sparse') or user_id >= self.liked_sparse.shape[0]:
            return None
            
        # Lấy danh sách phim đã liked
        start_idx = self.liked_sparse.indptr[user_id]
        end_idx = self.liked_sparse.indptr[user_id+1]
        liked = self.liked_sparse.indices[start_idx:end_idx]
        
        if len(liked) == 0:
            return None
            
        liked_indices = [self.movie_id_to_index[mid] for mid in liked if mid in self.movie_id_to_index]
        if not liked_indices:
            return None
            
        # Tính similarity của tất cả các phim so với các phim user đã thích
        tfidf_liked = self.tfidf_matrix[liked_indices]
        sim_matrix = cosine_similarity(self.tfidf_matrix, tfidf_liked)
        
        # Tính điểm tổng hợp (average similarity với các phim đã thích)
        scores = sim_matrix.mean(axis=1)
        
        # Loại bỏ các phim đã xem
        s_start = self.seen_sparse.indptr[user_id]
        s_end = self.seen_sparse.indptr[user_id+1]
        seen = self.seen_sparse.indices[s_start:s_end]
        
        seen_indices = [self.movie_id_to_index[mid] for mid in seen if mid in self.movie_id_to_index]
        scores[seen_indices] = -np.inf
        
        candidate_count = min(top_n * 5, len(scores))
        top_idx_unsorted = np.argpartition(scores, -candidate_count)[-candidate_count:]
        top_idx_sorted = top_idx_unsorted[np.argsort(scores[top_idx_unsorted])[::-1]]
        
        top_movie_indices = top_idx_sorted[:top_n]
        top_movie_ids = [self.index_to_movie_id[idx] for idx in top_movie_indices]
        top_scores = [float(scores[idx]) for idx in top_movie_indices]
        
        recs = pd.DataFrame({
            "movieId": top_movie_ids,
            "pred_score": top_scores
        }).merge(self.movies_df, on="movieId", how="left")
        
        columns_to_return = ["movieId", "title", "genres", "pred_score"]
        if "tmdbId" in recs.columns:
            columns_to_return.append("tmdbId")
            
        return recs[columns_to_return]


class HybridRecommender:
    def __init__(self, svd_model, cb_model, svd_weight=0.7):
        self.svd_model = svd_model
        self.cb_model = cb_model
        self.svd_weight = svd_weight
        self.cb_weight = 1.0 - svd_weight

    def recommend(self, user_id, top_n=10):
        # Lấy một không gian lớn hơn top_n để tìm điểm giao thoa (VD: top 500)
        svd_recs = self.svd_model.recommend(user_id, top_n=500)
        cb_recs = self.cb_model.recommend(user_id, top_n=500)

        # Xử lý trường hợp user không có data cho một trong hai thuật toán
        if svd_recs is None and cb_recs is None:
            return None
        elif svd_recs is None:
            return cb_recs.head(top_n)
        elif cb_recs is None:
            return svd_recs.head(top_n)

        # Trích xuất dữ liệu và đổi tên cột để merge
        df_svd = svd_recs[["movieId", "pred_score"]].rename(columns={"pred_score": "svd_score"})
        df_cb = cb_recs[["movieId", "pred_score"]].rename(columns={"pred_score": "cb_score"})

        # Merge outer để giữ lại toàn bộ phim có mặt trong ít nhất 1 list
        merged = pd.merge(df_svd, df_cb, on="movieId", how="outer")

        # Điền các điểm bị thiếu bằng min score của thuật toán đó
        min_svd = merged["svd_score"].min()
        min_cb = merged["cb_score"].min()
        merged["svd_score"] = merged["svd_score"].fillna(min_svd)
        merged["cb_score"] = merged["cb_score"].fillna(min_cb)

        # Min-Max Scaling về khoảng [0, 1] để cộng cho công bằng
        max_svd = merged["svd_score"].max()
        max_cb = merged["cb_score"].max()

        if max_svd > min_svd:
            merged["svd_norm"] = (merged["svd_score"] - min_svd) / (max_svd - min_svd)
        else:
            merged["svd_norm"] = 0.5

        if max_cb > min_cb:
            merged["cb_norm"] = (merged["cb_score"] - min_cb) / (max_cb - min_cb)
        else:
            merged["cb_norm"] = 0.5

        # Tính điểm lai ghép
        merged["pred_score"] = (self.svd_weight * merged["svd_norm"]) + (self.cb_weight * merged["cb_norm"])

        # Sắp xếp và lấy Top N
        final_recs = merged.sort_values(by="pred_score", ascending=False).head(top_n)

        # Điền lại các trường metadata bằng cách kết nối với movies_df có trong svd_model
        final_recs = final_recs[["movieId", "pred_score"]].merge(self.svd_model.movies_df, on="movieId", how="left")

        columns_to_return = ["movieId", "title", "genres", "pred_score"]
        if "tmdbId" in final_recs.columns:
            columns_to_return.append("tmdbId")

        return final_recs[columns_to_return]