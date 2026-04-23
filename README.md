# MovieLen-100K-AI-model

Dự án xây dựng hệ thống gợi ý phim sử dụng bộ dữ liệu MovieLens, có thể phục vụ qua giao diện ứng dụng hoặc API.

## Mục tiêu dự án

* Huấn luyện hoặc sử dụng mô hình gợi ý phim đã lưu sẵn
* Đọc dữ liệu từ thư mục `data/`
* Nạp mô hình từ thư mục `artifacts/`
* Chạy ứng dụng để dự đoán hoặc gợi ý phim cho người dùng

## Cấu trúc thư mục đề xuất

```bash
MovieLen-100K-AI-model/
├── data/
│   ├── ratings.csv
│   ├── tags.csv
│   └── ...
├── artifacts/
│   ├── recommender.joblib
│   └── ...
├── README.md
├── requirements.txt
└── ... các file mã nguồn khác ...
```

## Yêu cầu hệ thống

* Python 3.10 hoặc 3.11
* Git
* pip
* Khuyến nghị dùng môi trường ảo `venv`

## 1. Clone project

```bash
git clone https://github.com/truonglam003/MovieLen-100K-AI-model.git
cd MovieLen-100K-AI-model
```

## 2. Tạo và kích hoạt môi trường ảo

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Windows (CMD)

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Cài thư viện

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Chuẩn bị dữ liệu và model

Đảm bảo các file sau tồn tại đúng đường dẫn:

* `data/ratings.csv`
* `data/tags.csv`
* `artifacts/recommender.joblib`

Nếu bạn không có các file lớn này đúng vị trí, hãy tải chúng về và đặt đúng vị trí như trên trước khi chạy project.

## 5. Chạy project

Vì repo có cả `streamlit`, `fastapi` và `uvicorn`, dự án có thể đang hỗ trợ một trong hai cách chạy sau.

### Cách A: Chạy giao diện Streamlit

Nếu project có file như `app.py`, `streamlit_app.py` hoặc tên tương tự:

```bash
streamlit run app.py
```

Nếu file chính không phải `app.py`, hãy thay bằng tên file thật của bạn. Ví dụ:

```bash
streamlit run streamlit_app.py
```

Sau khi chạy, mở trình duyệt theo địa chỉ được terminal in ra, thường là:

```text
http://localhost:8501
```

### Cách B: Chạy API bằng FastAPI

Nếu project có file API như `main.py` chứa đối tượng `app = FastAPI()`:

```bash
uvicorn main:app --reload
```

Nếu file API có tên khác, thay `main` bằng tên file tương ứng. Ví dụ nếu file là `api.py`:

```bash
uvicorn api:app --reload
```

Sau khi chạy, truy cập:

* API: `http://127.0.0.1:8000`
* Swagger Docs: `http://127.0.0.1:8000/docs`

## 6. Cài lại môi trường trên máy khác

Khi clone project sang máy khác, bạn **không cần copy thư mục `venv/`**. Chỉ cần:

```bash
python -m venv venv
```

Kích hoạt môi trường ảo, rồi chạy:

```bash
pip install -r requirements.txt
```

## 7. Kiểm tra nhanh sau khi cài

Bạn có thể kiểm tra các thư viện chính đã cài thành công bằng lệnh:

```bash
python -c "import pandas, numpy, scipy, sklearn, streamlit, joblib, requests, fastapi, uvicorn; print('OK')"
```

Nếu terminal in ra `OK` thì môi trường đã sẵn sàng.

## 8. Một số lỗi thường gặp

### Không tìm thấy file dữ liệu hoặc model

Nguyên nhân:

* Thiếu file trong `data/` hoặc `artifacts/`
* Sai đường dẫn trong code

Cách xử lý:

* Kiểm tra lại tên file
* Kiểm tra đúng thư mục
* Nếu dùng Git LFS, hãy chắc rằng file đã được tải đầy đủ

### Không chạy được `streamlit` hoặc `uvicorn`

Cách xử lý:

```bash
pip install -r requirements.txt
```

Sau đó thử lại.

### Lỗi do khác phiên bản Python

Khuyến nghị dùng:

* Python 3.10
* hoặc Python 3.11

## 9. Gợi ý cho người dùng mới

Thứ tự chạy an toàn nhất:

1. Clone repo
2. Tạo `venv`
3. Cài `requirements.txt`
4. Đặt dữ liệu và model vào đúng thư mục
5. Chạy Streamlit hoặc FastAPI
