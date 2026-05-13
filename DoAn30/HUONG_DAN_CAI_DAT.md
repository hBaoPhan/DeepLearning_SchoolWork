# Hướng Dẫn Cài Đặt Môi Trường Python

Tài liệu này hướng dẫn bạn cách tạo một môi trường Python mới và cài đặt các thư viện cần thiết để chạy file Jupyter Notebook `Đồ_Án_30_DeepLearning (4).ipynb` và ứng dụng `app.py`.

## 1. Yêu cầu hệ thống
- Đã cài đặt **Python 3.9 - 3.11** (Khuyên dùng 3.10 để tương thích tốt nhất với TensorFlow).
- Hoặc đã cài đặt **Anaconda/Miniconda**.

---


## 2.  Sử dụng Anaconda/Miniconda

Mở **Anaconda Prompt** và chạy các lệnh sau:

### Bước 1: Tạo môi trường mới
```bash
conda create -n doan30_env python=3.10 -y
```

### Bước 2: Kích hoạt môi trường
```bash
conda activate doan30_env
```

### Bước 3: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

---

## 3. Cấu hình để chạy Notebook trong VS Code hoặc Jupyter

Sau khi đã cài đặt xong, bạn cần cài đặt kernel để Notebook có thể nhận diện môi trường mới:

```bash
python -m ipykernel install --user --name doan30_env --display-name "Python (DoAn30)"
```

**Lưu ý:** Khi mở file `.ipynb` trong VS Code, hãy nhấn vào nút **"Select Kernel"** ở góc trên bên phải và chọn **"Python (DoAn30)"** hoặc môi trường bạn vừa tạo.

---

## 5. Chạy ứng dụng Streamlit (`app.py`)

Nếu bạn muốn chạy demo giao diện thực tế:
```bash
streamlit run app.py
```

---

## Danh sách các thư viện chính đã cài đặt:
- `tensorflow`: Thư viện Deep Learning để huấn luyện mô hình LSTM.
- `pandas`, `numpy`: Xử lý dữ liệu chuỗi thời gian.
- `scikit-learn`: Tiền xử lý dữ liệu (MinMaxScaler) và đánh giá mô hình.
- `matplotlib`: Vẽ biểu đồ phân tích.
- `streamlit`: Giao diện web mô phỏng Real-time.
- `ipykernel`: Hỗ trợ chạy Notebook trên môi trường ảo.
