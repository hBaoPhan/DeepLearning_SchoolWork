# 🚀 Giải Thích Chi Tiết Ứng Dụng `app.py` (Real-time Anomaly Detection)

Tài liệu này giải thích chi tiết luồng hoạt động và từng đoạn code quan trọng trong ứng dụng Streamlit `app.py`. Đây là ứng dụng mô phỏng hệ thống cảnh báo bất thường thời gian thực (Real-time Streaming) dựa trên mô hình Seq2Seq LSTM.

---

## 🌊 Luồng Hoạt Động Tổng Quan (Execution Flow)

Ứng dụng hoạt động theo mô hình **Event-driven (Hướng sự kiện)** kết hợp với **Vòng lặp Mô phỏng (Simulation Loop)**:
1. **Khởi tạo:** Nạp thư viện, cấu hình giao diện UI, tải file CSV và khởi tạo Mô hình AI vào bộ nhớ.
2. **Khởi tạo Trạng thái (Session State):** Streamlit sẽ chạy lại toàn bộ script mỗi khi người dùng tương tác. Do đó, cần `st.session_state` để lưu lại toàn bộ lịch sử điểm vẽ (data, preds, scores, thresholds) qua từng vòng lặp mà không bị mất đi.
3. **Tính chuẩn cố định (Global Fixed STD):** Ứng dụng quét 100 dòng đầu tiên của dữ liệu để tìm ra độ lệch chuẩn lý tưởng (`fixed_std`).
4. **Vòng lặp Streaming:** Khi ấn nút "▶️ Bắt đầu Chạy", ứng dụng lọt vào một vòng lặp `for`. Mỗi bước lặp mô phỏng 1 giây thời gian thực:
   - Cắt 100 điểm dữ liệu quá khứ ném vào AI để dự báo quỹ đạo.
   - Trích xuất điểm hiện tại để đối chiếu với AI $\rightarrow$ Ra được sai số (MAE).
   - Đưa sai số này so sánh với **Ngưỡng Động (Adaptive Threshold)**. Nếu vượt ngưỡng $\rightarrow$ Kích hoạt báo động.
   - Vẽ lại đồ thị, ngủ 1 chút (`time.sleep`) rồi sang bước tiếp theo.

---

## 💻 Giải Thích Chi Tiết Từng Phần Code

### 1. Khởi tạo Trạng Thái (Session State)
```python
if 'history' not in st.session_state:
    st.session_state.history = {
        'data': [],          # Dữ liệu thực tế từng bước
        'preds': [],         # Dữ liệu AI dự báo
        'scores': [],        # Điểm sai số MAE
        'thresholds': [],    # Ngưỡng cảnh báo động
        'anomalies': [],     # Kết quả (1: Lỗi, 0: Bình thường)
        'true_labels': [],   # Nhãn gốc (Ground Truth)
        'step': 0            # Vị trí dòng hiện tại trong file CSV
    }
```
* **Mục đích:** Khác với Python thông thường, Streamlit tự reset mọi biến mỗi khi ấn nút. `st.session_state` giống như một cái "túi thần kỳ" giữ lại dữ liệu lịch sử để biểu đồ vẽ liên tục mà không bị xóa mờ.

### 2. Hàm Cập Nhật Đồ Thị (`update_charts`)
```python
def update_charts(...):
    plot_len = 100
    display_data = history['data'][-plot_len:]
    ...
```
* **Mục đích:** Hàm này dùng `matplotlib` vẽ 2 biểu đồ.
* Nó chỉ trích xuất **đúng 100 điểm cuối cùng** (`[-plot_len:]`) trong lịch sử để vẽ, nhằm tạo hiệu ứng "Cửa sổ trượt" (Sliding window visual) chạy ngang màn hình, giống hệt màn hình nhịp tim trong bệnh viện.
* Logic đánh dấu: Dấu chấm đỏ (`^`) cho lỗi thực tế, dấu gạch chéo nâu (`x`) cho AI phát hiện lỗi, và hàm `fill_between` để tô màu vùng MAE vượt Ngưỡng (vùng nguy hiểm).

### 3. Nạp Dữ Liệu và Tính Độ Lệch Chuẩn Chuẩn (Global Fixed STD)
```python
df = pd.read_csv(...)
full_data_scaled = scaler.fit_transform(df[['value']])

# Calculate Global Fixed STD ONCE per file to avoid recalculating
if 'global_fixed_std' not in st.session_state ...:
    # Quét 100 cửa sổ đầu tiên của file CSV
    # ...
    st.session_state.global_fixed_std = np.std(fs) 
```
* **Mục đích:** Đọc nguyên file dữ liệu. Sau đó, nó sẽ tính toán **Độ lệch chuẩn cố định (Fixed STD)** dựa trên 100 điểm đầu tiên của file. 
* **Tại sao?** 100 điểm đầu thường là dữ liệu Bình thường (chuẩn). Ta lấy độ dao động của vùng này làm biên độ cứng. Việc này giúp Ngưỡng cảnh báo luôn giữ được sự rắn rỏi, không bị "phình to" đánh lừa bởi các dữ liệu nhiễu diễn ra liên tiếp trong tương lai (Hybrid Threshold Strategy).

### 4. Tính Năng "Tua Nhanh" Thông Minh (Fast-Forward Pre-calculation)
```python
if jump_step != st.session_state.history['step']:
    st.session_state.history['step'] = jump_step
    # Tính toán lùi 150 bước trong quá khứ...
    batch_preds = model.predict(windows, verbose=0)
    ...
```
* **Mục đích:** Nếu bạn tua thẳng đến bước thứ 1000, thay vì hệ thống mất phương hướng vì lịch sử trống trơn (làm gãy đường `rolling_mean`), nó sẽ ngầm **tính toán bù** (Batch processing) cho 150 bước từ khoảng 850 đến 999.
* Kết quả là khi tới bước 1000, đồ thị đã có sẵn đà mượt mà của 150 điểm trước đó. Thao tác chớp nhoáng vì dùng Batch Prediction của mô hình Keras.

### 5. Vòng Lặp Streaming & Tránh Rò Rỉ Dữ Liệu (No Future Peeking)
```python
for i in range(start_idx, len(df) - FORECAST_HORIZON):
    ...
    # 1. Trích xuất cửa sổ 100 điểm QUÁ KHỨ
    window = full_data_scaled[i-LOOKBACK:i].reshape(1, LOOKBACK, 1)
    
    # 2. AI Dự đoán tương lai
    pred = model.predict(window, verbose=0)
    
    # 3. Tính MAE THỜI GIAN THỰC (Chỉ dùng điểm HIỆN TẠI)
    score = np.abs(current_val - pred[0, 0, 0])
```
* **Ý nghĩa:** Đây là trái tim của hệ thống Real-time. AI nhận vào mảng `window` chứa 100 giây quá khứ và phun ra dự báo.
* **Quy tắc Vàng (Real-time Rules):** Biến `score` tính sai số (MAE) tuyệt đối chỉ dựa trên `current_val` (giá trị thực sự xảy ra ở ngay giây `i`) và `pred[0, 0, 0]` (dự đoán đầu tiên của AI). Hệ thống tuân thủ chặt chẽ nguyên lý **"Không ăn gian nhìn trước dữ liệu tương lai" (No Data Leakage)**.

### 6. Ngưỡng Động Linh Hoạt (Adaptive Threshold)
```python
    # Lấy cửa sổ điểm số cách đây 50 bước
    window_scores = history_scores.shift(SHIFT_STEPS).tail(ROLLING_WINDOW)
    rolling_mean = window_scores.mean()

    # Tính Ngưỡng tổng hợp
    threshold = rolling_mean + (K_STD * st.session_state.global_fixed_std) + epsilon
```
* **Rolling Mean Trễ 50 Bước (`SHIFT_STEPS`):** Trung bình của dữ liệu (điểm mỏ neo của Ngưỡng) không tính ở ngay sát hiện tại, mà lấy của 50 bước trước. Nhờ vậy, nếu xảy ra tấn công dồn dập ở hiện tại, đường trung bình sẽ không bị bẻ cong theo ngay lập tức $\rightarrow$ Hệ thống bắt lỗi nhạy bén hơn.
* **Ngưỡng cuối cùng (`threshold`):** Bằng (Trung bình cuộn) + (`K_STD` nhân với Biên độ chuẩn ban đầu). Khi `K_STD` trên giao diện càng lớn, Threshold càng cao $\rightarrow$ Ít cảnh báo ảo (nhưng có thể bỏ sót lỗi nhỏ).

### 7. Phân Loại và Cập Nhật Giao Diện
```python
    is_anomaly = 1 if score > threshold else 0
    # ... lưu vào history ...
    st.pyplot(fig_data) # Vẽ đồ thị
    time.sleep(SPEED)   # Chờ một lát (mô phỏng thời gian)
```
* **Mục đích:** Đơn giản là so sánh MAE với Ngưỡng. Lớn hơn là Cảnh Báo, Nhỏ hơn là Bình Thường. Giao diện được ép vẽ lại, và `time.sleep` tạo cảm giác chờ đợi thời gian thực chảy trôi của dòng suối dữ liệu.

---
**Tổng kết:** `app.py` là một bản giao hưởng kết hợp giữa sức mạnh trí tuệ nhân tạo (Mô hình Keras), luồng xử lý dữ liệu chống Data Leakage chuẩn mực, và một hệ thống tính Ngưỡng cuộn linh hoạt vững chãi.
