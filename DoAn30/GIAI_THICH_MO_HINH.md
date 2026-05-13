# 🛡️ Giải Thích Chi Tiết Dự Án Anomaly Detection (Seq2Seq LSTM)

Tài liệu này tổng hợp toàn bộ kiến thức, logic cốt lõi và cơ chế hoạt động của dự án Phát hiện Bất thường Lưu lượng Mạng (Anomaly Detection), bao gồm file Notebook huấn luyện và ứng dụng Live Demo `app.py`.

---

## 1. Kiến trúc Mô hình (Seq2Seq LSTM - 10-step Forecasting)

Mô hình cốt lõi của dự án không phải là dự báo 1 bước (1-step) cơ bản, mà sử dụng cấu trúc **Sequence-to-Sequence (Seq2Seq)** dự báo nhiều bước cùng lúc.
*   **Lookback = 100**: Nhìn lại 100 điểm dữ liệu trong quá khứ.
*   **Forecast Horizon = 10**: Dự báo một quỹ đạo gồm 10 điểm trong tương lai.

### Luồng xử lý qua 4 khối:
1. **Encoder (Khối Mã hóa):** Mạng LSTM đọc 100 điểm quá khứ, trích xuất quy luật và nén toàn bộ thành 1 trạng thái cốt lõi duy nhất gọi là **Context Vector** (`return_sequences=False`).
2. **Bridge (Khối Cầu nối):** Lớp `RepeatVector(10)` nhân bản Context Vector này lên 10 lần, tạo "bối cảnh" để nhắc bài cho Decoder ở mỗi bước tương lai.
3. **Decoder (Khối Giải mã):** Lớp LSTM thứ hai nhận 10 bản sao này và bung ra thành các trạng thái dự báo (`return_sequences=True`).
4. **Output (Khối Đầu ra):** Lớp `TimeDistributed(Dense(1))` áp dụng hàm tuyến tính ĐỘC LẬP lên từng bước, xuất ra chính xác **10 con số tương lai**.

### Tại sao lại là 10 bước (Sự đánh đổi - Trade-off):
Dự báo xa 10 bước giúp làm mượt các điểm nhiễu cục bộ (tránh cảnh báo cháy giả / False Positive như mô hình 1-step). Việc không chọn dự báo quá xa (50-100 bước) giúp mô hình không bị mất độ chính xác (tránh bỏ lọt lỗi / False Negative). Con số 10 bước (tỉ lệ 10% của lookback) là **điểm cân bằng lý tưởng (Sweet Spot)**.

---

## 2. Cơ Chế Bắt Lỗi (MAE & Threshold)

**AI KHÔNG dự đoán thời điểm xảy ra lỗi**, vì lỗi là ngẫu nhiên. Thay vào đó, AI dự báo **quỹ đạo của sự Bình thường**.

### Sai số MAE (Mean Absolute Error)
Khi dữ liệu thực tế (`target`) tới, hệ thống tính toán sai số so với những gì AI đã dự báo (`pred`):
`score = np.mean(np.abs(target - pred))`
*   Nếu `score` nhỏ: Dữ liệu thực tế đi đúng kịch bản của AI $\rightarrow$ Hệ thống Bình thường.
*   Nếu `score` lớn: Dữ liệu thực tế bị chệch quỹ đạo (do hacker, rớt mạng, v.v.) $\rightarrow$ Xảy ra Bất thường.

### Ngưỡng Cảnh Báo (Threshold)
Sự bất thường được kích hoạt (còi báo động réo) khi MAE vượt qua một ranh giới (Threshold).
Có 2 phương pháp tính Threshold nổi bật trong dự án:
1. **Adaptive Threshold (Ngưỡng cuộn động - `rolling_std`):**
   * Tự động điều chỉnh theo môi trường thời gian thực.
   * *Nhược điểm:* Dễ bị "phình to" ngưỡng nếu bị tấn công tàng hình kéo dài, dẫn đến đánh lừa hệ thống.
2. **Hybrid Threshold (Kết hợp Validation Fixed STD - `val_std_fixed`):**
   * Lấy `val_std_fixed` (Độ lệch chuẩn tĩnh) từ tập Validation sạch để khóa cứng biên độ dao động.
   * Đây là phương pháp **Best Practice**, giữ cho ngưỡng cứng rắn không bị nới lỏng bởi hacker, nhưng vẫn chạy trượt linh hoạt nhờ `rolling_mean`.

---

## 3. Cấu Trúc Tensor (Đầu Ra Của Mô Hình)

Khi debug hoặc xem output của mô hình, bạn sẽ bắt gặp 2 dạng cấu trúc Mảng (Tensor):

### 3.1. Mảng 2 Chiều `[[ ]]` - Shape: `(10, 1)`
*   **Ví dụ:** `[[4650.22], [1708.46], ...]`
*   **Ý nghĩa:** Đây là kết quả dự báo tại **MỘT thời điểm duy nhất**. Gồm 10 giá trị tương lai của 1 cửa sổ trượt. Bạn sẽ gặp mảng này trong luồng chạy thời gian thực (Live Streaming) của file `app.py`.

### 3.2. Mảng 3 Chiều `[[[ ]]]` - Shape: `(N, 10, 1)`
*   **Ví dụ:** `[[[-151.74], ...], [[-264.31], ...]]`
*   **Ý nghĩa:** Đây là kết quả dự báo của **TOÀN BỘ Tập dữ liệu** (Batch Processing) gồm N cửa sổ trượt. Bạn sẽ gặp nó khi chạy lệnh `model.predict(X_val)` trong Jupyter Notebook.

---

## 4. Hướng Dẫn Demo Thực Tế (Live Presentation)

Ứng dụng `app.py` viết bằng Streamlit là một sản phẩm hoàn thiện để báo cáo và trình diễn trước hội đồng/khách hàng.

**Kịch bản Demo khuyên dùng:**
1. **Chế độ Giám sát (Normal Monitoring):** 
   * Bật hệ thống và cho chạy file CSV bình thường. 
   * Mục tiêu: Cho người xem thấy Ngưỡng động bám sát đường MAE một cách mượt mà, không kích hoạt báo động giả (False Positive).
2. **Kiểm thử Sự cố (Stress Test / Anomaly Injection):** 
   * Nhấn nút **"💥 Chèn Lỗi"**. 
   * Mục tiêu: Giả lập cuộc tấn công DDoS/sập nguồn. Chứng minh hệ thống bắt ngay lập tức tại giây đầu tiên khi dữ liệu chệch quỹ đạo (Không có độ trễ dài).
3. **Tùy chỉnh Độ nhạy (Sensitivity Tuning):** 
   * Kéo thanh trượt `K_STD`. 
   * Mục tiêu: Chứng minh tính linh hoạt của hệ thống. Kéo thấp để thắt chặt an ninh (nhạy cảm cao), kéo cao để tránh phiền nhiễu cho doanh nghiệp.

---
*Tài liệu được tổng hợp dành riêng cho việc bảo vệ đồ án/trình bày dự án.*
