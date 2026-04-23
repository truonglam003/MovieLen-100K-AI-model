import os

import joblib
import pandas as pd
import requests
import streamlit as st

from config import POPULARITY_MIN_RATINGS, TOP_N, TMDB_API_KEY
from models import popularity_recommend, HybridRecommender
from train_model import train_and_save


ARTIFACT_PATH = os.environ.get("TTCS_ARTIFACT_PATH", "artifacts/recommender.joblib")


@st.cache_data(show_spinner=False)
def fetch_movie_poster(tmdb_id, api_key):
    if not tmdb_id or pd.isna(tmdb_id) or not api_key:
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={api_key}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        pass
    return None

def display_movie_grid(df, api_key):
    if df.empty:
        st.info("Không có dữ liệu để hiển thị.")
        return

    cols_per_row = 5
    for i in range(0, len(df), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(df):
                row = df.iloc[i + j]
                tmdb_id = row.get("tmdbId")
                title = row["title"]
                genres = row["genres"]
                
                score_str = ""
                if "pred_score" in row:
                    score_str = f"⭐ {row['pred_score']:.2f}"
                elif "score" in row and "avg_rating" in row:
                    score_str = f"⭐ {row['avg_rating']:.2f} ({int(row.get('rating_count', 0))} lượt)"

                poster_url = fetch_movie_poster(tmdb_id, api_key)
                
                with col:
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    else:
                        st.image("https://via.placeholder.com/500x750?text=No+Poster", use_container_width=True)
                    
                    st.markdown(f"**{title}**")
                    st.caption(genres)
                    if score_str:
                        st.markdown(score_str)


@st.cache_resource
def load_artifacts(path: str):
    return joblib.load(path)


def ensure_artifacts():
    if os.path.exists(ARTIFACT_PATH):
        return True

    st.warning("Chưa có model đã train. Hãy train 1 lần để app load nhanh.")
    if st.button("Train & save model", type="primary"):
        with st.spinner("Đang train model (lần đầu sẽ hơi lâu)..."):
            train_and_save(out_path=ARTIFACT_PATH)
        st.success("Train xong. Bạn có thể dùng ngay.")
        st.cache_resource.clear()
        return True

    return False


def main():
    st.set_page_config(page_title="Movie Recommender", layout="wide")
    st.title("Movie Recommender (Streamlit)")

    if not ensure_artifacts():
        st.stop()

    artifacts = load_artifacts(ARTIFACT_PATH)
    model = artifacts["model"]
    cb_model = artifacts.get("cb_model")
    popular_df = artifacts.get("popular_df")
    train_df = artifacts.get("train_df")
    movies_df = artifacts.get("movies_df")
    meta = artifacts.get("meta", {})
    
    # Khởi tạo HybridRecommender on the fly (0.7 SVD + 0.3 CB)
    hybrid_model = None
    if cb_model is not None:
        hybrid_model = HybridRecommender(model, cb_model, svd_weight=0.7)

    with st.sidebar:
        st.subheader("Cấu hình")
        api_key_input = st.text_input("TMDB API Key", value=TMDB_API_KEY, type="password", help="Dùng để tải ảnh bìa phim")
        top_n = st.slider("Top-N", min_value=1, max_value=50, value=int(meta.get("top_n_default", TOP_N)))
        min_ratings = st.slider(
            "Min ratings (popularity fallback)",
            min_value=1,
            max_value=500,
            value=int(meta.get("popularity_min_ratings_default", POPULARITY_MIN_RATINGS)),
        )
        
        algorithm = st.selectbox("Thuật toán", ["SVD", "Content-Based", "Hybrid"])

        st.divider()
        st.caption("Thông tin model")
        st.write(
            {
                "n_components": meta.get("n_components"),
                "explained_variance_sum": meta.get("explained_variance_sum"),
                "min_user_ratings": meta.get("min_user_ratings"),
                "min_movie_ratings": meta.get("min_movie_ratings"),
            }
        )

    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        st.subheader("Nhập user id")
        user_input = st.text_input("User ID", value="", placeholder="Ví dụ: 1")
        run = st.button("Gợi ý", type="primary", use_container_width=True)

        st.caption(f"Artifact: `{ARTIFACT_PATH}`")

    with col2:
        st.subheader("Kết quả")

        if run:
            user_input = user_input.strip()
            if not user_input:
                st.info("Hãy nhập user id.")
                st.stop()

            try:
                user_id = int(user_input)
            except ValueError:
                st.error("User ID không hợp lệ. Vui lòng nhập số nguyên.")
                st.stop()

            if algorithm == "SVD":
                recs = model.recommend(user_id, top_n=top_n)
            elif algorithm == "Content-Based":
                if cb_model is not None:
                    recs = cb_model.recommend(user_id, top_n=top_n)
                else:
                    st.error("Mô hình Content-Based chưa được train. Hãy bấm Train & save model.")
                    st.stop()
            elif algorithm == "Hybrid":
                if hybrid_model is not None:
                    recs = hybrid_model.recommend(user_id, top_n=top_n)
                else:
                    st.error("Mô hình Hybrid chưa sẵn sàng do thiếu Content-Based.")
                    st.stop()

            if recs is None:
                st.warning(f"User {user_id} không có trong dữ liệu. Hiển thị phim phổ biến.")

                if train_df is not None and movies_df is not None:
                    pop = popularity_recommend(
                        train_df,
                        movies_df,
                        top_n=top_n,
                        min_ratings=min_ratings,
                    )
                    show_df = pop
                elif isinstance(popular_df, pd.DataFrame):
                    show_df = popular_df.head(top_n)
                else:
                    st.error("Không có dữ liệu fallback popularity trong artifacts.")
                    st.stop()

                display_movie_grid(show_df, api_key_input)
            else:
                st.success(f"Top {top_n} phim gợi ý cho user {user_id}")
                display_movie_grid(recs, api_key_input)

        else:
            st.info("Nhập user id rồi bấm **Gợi ý**.")


if __name__ == "__main__":
    main()

