# Goal Description

Nâng cấp giao diện dự án dể hiển thị Poster phim bằng cách dùng ID từ file `links.csv` và gọi API của TMDB (The Movie Database). Điều này sẽ giúp ứng dụng Streamlit trông trực quan và cao cấp (giống Netflix) hơn thay vì chỉ hiện bảng danh sách chữ.

> [!IMPORTANT]
> **Yêu cầu đối với Bạn:** Để giao diện hiện được ảnh, bạn cần tạo tài khoản trên [themoviedb.org](https://www.themoviedb.org/) và lấy một **API Key** (hoàn toàn miễn phí).

## User Review Required

Tôi sẽ thêm tính năng nhập API Key trực tiếp trên giao diện web (hoặc lưu trong code `config.py` tuỳ bạn chọn) để ứng dụng có thể kết nối với hệ thống TMDB tải cấu trúc ảnh về.

## Proposed Changes

### Configuration & Environment
#### [MODIFY] [requirements.txt](file:///d:/TTCS/requirements.txt)
- Thêm thư viện `requests` để dùng cho việc gọi API tải dữ liệu ảnh.

#### [MODIFY] [config.py](file:///d:/TTCS/config.py)
- Thêm cấu hình đường dẫn `LINKS_PATH = "data/links.csv"`.
- Thêm biến `TMDB_API_KEY = ""` để bạn dễ dàng điền mã sau này.

---

### Data Processing Logic
#### [MODIFY] [data_loader.py](file:///d:/TTCS/data_loader.py)
- Cập nhật hàm `load_data` để đọc thêm file `links.csv`.
- Gộp (merge) cột `tmdbId` từ bảng `links` sang bảng bảng `movies_df` theo `movieId`.

#### [MODIFY] [train_model.py](file:///d:/TTCS/train_model.py)
- Nạp thêm `LINKS_PATH`.
- Đổi cách gọi sang `load_data(RATINGS_PATH, MOVIES_PATH, LINKS_PATH)` để khi bạn chạy file train, `tmdbId` sẽ luôn có sẵn trong file lưu `artifacts/recommender.joblib`.

---

### Web Interface & Display
#### [MODIFY] [streamlit_app.py](file:///d:/TTCS/streamlit_app.py)
- Xây dựng hàm `fetch_movie_poster(tmdb_id)` dùng thư viện `requests` để lấy ảnh bìa từ TMDB. Hàm này sẽ được dùng với `%st.cache_data` để load ảnh rất nhanh ở những lần sau.
- Viết lại giao diện phần hiển thị: Thay vì dùng bảng `st.dataframe` nhàm chán, tôi sẽ tạo một **lưới (grid)** để hiển thị hình ảnh phim cover thật đẹp mắt, có kèm thông tin tiêu đề và thể loại bên dưới mỗi ảnh.
- Bổ sung ô nhập "TMDB API Key" ở thanh công cục (Sidebar) để nếu bạn lỡ không viết API thay vào file config thì bạn vẫn điền trực tiếp trên màn hình web được.