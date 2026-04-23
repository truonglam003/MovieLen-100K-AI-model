# Kế hoạch Triển khai FastAPI Backend

Dựa trên sự đồng ý của bạn, chúng ta sẽ bắt đầu chuyển hóa dự án từ một bản thử nghiệm (Streamlit) sang một kiến trúc Client-Server chuyên nghiệp. Bước đầu tiên là xây dựng một **API Backend** bằng **FastAPI** để bọc các thuật toán AI lại.

## User Review Required

> [!IMPORTANT]
> - Mình sẽ cài đặt thêm thư viện `fastapi` và `uvicorn` (máy chủ web). Bạn có đồng ý cho phép mình tự động cài đặt các thư viện này vào môi trường ảo hiện tại không?
> - Ở giai đoạn này, API sẽ trả về dữ liệu phim dưới dạng JSON (bao gồm `movieId`, `title`, `genres`, `pred_score`, và `tmdbId`). Giao diện Frontend (React/Vue) sau này sẽ dùng `tmdbId` để tải ảnh bìa. Như vậy đã phù hợp với định hướng của bạn chưa?

## Proposed Changes

### 1. Cập nhật thư viện

#### [MODIFY] [requirements.txt](file:///d:/TTCS/requirements.txt)
- Thêm `fastapi`
- Thêm `uvicorn`
- Thêm `pydantic` (định dạng dữ liệu)

### 2. Xây dựng API Server chính

#### [NEW] [api.py](file:///d:/TTCS/api.py)
Tạo file server chạy FastAPI với các chức năng:
- **Khởi tạo và Load Model:** Tự động load file `artifacts/recommender.joblib` vào bộ nhớ RAM khi server khởi động (chỉ load 1 lần để tối ưu tốc độ). Tự động khởi tạo `HybridRecommender`.
- **Route `GET /`**: Endpoint kiểm tra sức khỏe của server (Health Check).
- **Route `GET /api/recommend`**: Endpoint chính nhận các tham số:
  - `user_id` (kiểu số nguyên).
  - `top_n` (mặc định 10).
  - `algo` (chọn giữa `svd`, `cb`, `hybrid` - mặc định là `hybrid`).
  - Xử lý logic gọi hàm AI tương ứng và trả về kết quả dưới định dạng chuỗi JSON.
  - Tự động Fallback về thuật toán Popularity nếu User ID không tồn tại.

## Verification Plan

### Automated Tests
- Chạy thử lệnh `uvicorn api:app --reload` để khởi động server.
- Sử dụng trình duyệt hoặc công cụ như Postman / cURL để gọi thử API: `http://localhost:8000/api/recommend?user_id=5&algo=hybrid`
- Kiểm tra xem kết quả JSON trả về có khớp với kết quả trên màn hình Streamlit mà chúng ta đã test trước đó hay không.

---
Bạn vui lòng kiểm tra kế hoạch trên. Nếu bạn đồng ý với các thay đổi và câu hỏi ở phần **User Review Required**, hãy báo cho mình biết để mình tiến hành cài đặt và viết code nhé!
