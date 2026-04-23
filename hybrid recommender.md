# Triển khai Hybrid Recommender (Lọc Lai Ghép)

Để nâng cấp dự án theo kiến trúc gần hơn với hệ thống thực tế (giống Netflix), mình đã tiếp tục bổ sung thuật toán **Hybrid Recommender** bằng cách kết hợp sức mạnh của Collaborative Filtering (SVD) và Content-Based (CB).

## Những thay đổi chính

### 1. Thuật toán lai ghép Hybrid (`models.py`)
- Mình đã tạo thêm class `HybridRecommender` nhận vào cả 2 model SVD và CB.
- Thuật toán thực hiện lấy Top 500 điểm dự đoán của mỗi mô hình (để có không gian giao thoa đủ lớn), sau đó thực hiện **Outer Merge** (Gộp dữ liệu). Những phim không được SVD hoặc CB chấm điểm sẽ bị "phạt" bằng cách lấy mức điểm thấp nhất (min) của hệ thống đó.
- Điểm từ 2 thuật toán có thang đo hoàn toàn khác nhau (SVD từ 1-5, CB từ 0-1). Nên mình đã dùng kỹ thuật **Min-Max Scaling** để ép cả 2 cột điểm về cùng một khoảng chuẩn hóa [0, 1].
- Điểm dự đoán cuối cùng (Final Score) được tính dựa trên tỷ lệ cố định bạn yêu cầu: 
  `pred_score = (0.7 * Điểm_SVD_chuẩn_hóa) + (0.3 * Điểm_CB_chuẩn_hóa)`

### 2. Tích hợp UI Thử nghiệm (`streamlit_app.py`)
- Thêm thuật toán **"Hybrid"** vào danh sách Selectbox ở thanh bên.
- Điều đặc biệt là thuật toán Hybrid được **khởi tạo tự động (on the fly)** trong file giao diện bằng cách truyền 2 model đã được load từ bộ nhớ vào. Vì vậy, bạn **không cần phải train lại model**. File `recommender.joblib` hiện tại vẫn sử dụng tốt mà không bị tăng dung lượng hay phá vỡ cấu trúc cũ.

## Kết quả Verification
Mình đã viết test script kiểm tra thuật toán Hybrid trên User `5`. Kết quả trả ra rất hợp lý:

> Các bộ phim bom tấn như *Terminator 2*, *Titanic*, *Independence Day* vốn được SVD đánh giá cao giờ đã được sắp xếp lại dựa trên cả yếu tố Content-Based (người này thích thể loại Hành động/Kịch tính). Điểm số `pred_score` hiện tại nằm trong khoảng `[0, 1]` chuẩn theo thang đo Mix-Max Scaling.