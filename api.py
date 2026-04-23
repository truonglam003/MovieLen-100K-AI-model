import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
import uvicorn

from models import popularity_recommend
from config import TOP_N, POPULARITY_MIN_RATINGS

ARTIFACT_PATH = os.environ.get("TTCS_ARTIFACT_PATH", "artifacts/recommender.joblib")

# Biến toàn cục để lưu model sau khi load
app_data = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chạy khi khởi động Server
    if os.path.exists(ARTIFACT_PATH):
        print(f"Loading model artifacts from {ARTIFACT_PATH}...")
        artifacts = joblib.load(ARTIFACT_PATH)
        app_data["model"] = artifacts.get("model")
        app_data["popular_df"] = artifacts.get("popular_df")
        app_data["train_df"] = artifacts.get("train_df")
        app_data["movies_df"] = artifacts.get("movies_df")
        app_data["meta"] = artifacts.get("meta", {})
        print("Model loaded successfully!")
    else:
        print(f"Warning: Artifact not found at {ARTIFACT_PATH}. Please train the model first.")
        app_data["model"] = None
        
    yield  # Nhường quyền cho app chạy
    
    # Chạy khi tắt Server
    app_data.clear()

# Khởi tạo ứng dụng FastAPI với lifespan event
app = FastAPI(title="Movie Recommender API", lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "Welcome to Movie Recommender API. Go to /docs to test endpoints."}

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": app_data.get("model") is not None}

@app.get("/api/recommend", tags=["Recommendations"])
def recommend_movies(
    user_id: int = Query(..., description="ID của người dùng cần gợi ý phim"),
    top_n: int = Query(TOP_N, ge=1, le=50, description="Số lượng phim cần gợi ý")
):
    """
    Gợi ý phim cho người dùng dựa trên thuật toán Collaborative Filtering (SVD).
    Nếu là người dùng mới (chưa có trong dữ liệu huấn luyện), sẽ fallback về Popularity-based.
    """
    model = app_data.get("model")
    train_df = app_data.get("train_df")
    movies_df = app_data.get("movies_df")
    popular_df = app_data.get("popular_df")
    meta = app_data.get("meta", {})

    if not model:
        raise HTTPException(status_code=503, detail="Model chưa được huấn luyện hoặc không tìm thấy artifact.")

    # Thử gợi ý bằng mô hình SVD
    recs = model.recommend(user_id, top_n=top_n)

    if recs is None:
        # User mới (Cold-start) -> dùng Fallback popularity
        min_ratings = meta.get("popularity_min_ratings_default", POPULARITY_MIN_RATINGS)
        
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
            raise HTTPException(status_code=500, detail="Không có dữ liệu Fallback Popularity.")
        
        # Chuyển đổi DataFrame thành list các dictionary để FastAPI trả về dạng JSON
        show_df["is_fallback"] = True
        
        # Đảm bảo handle NaN values cho JSON
        show_df = show_df.fillna("") 
        return show_df.to_dict(orient="records")

    # Nếu thành công (User cũ)
    recs["is_fallback"] = False
    recs = recs.fillna("") 
    return recs.to_dict(orient="records")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
