import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf

# --- Cấu hình Trang ---
st.set_page_config(
    page_title="Anomaly Detection Live Demo",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- Khởi tạo Session State ---
if 'history' not in st.session_state:
    st.session_state.history = {
        'data': [], 
        'preds': [],
        'scores': [], 
        'thresholds': [], 
        'anomalies': [], 
        'true_labels': [],
        'step': 0
    }
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'inject_target_step' not in st.session_state:
    st.session_state.inject_target_step = -1

# --- Hàm hỗ trợ ---
@st.cache_resource
def load_anomaly_model():
    try:
        return tf.keras.models.load_model('checkpoint/seq2seq_anomaly_model.keras')
    except:
        return None

def reset_simulation():
    st.session_state.history = {'data': [], 'preds': [], 'scores': [], 'thresholds': [], 'anomalies': [], 'true_labels': [], 'step': 0}
    st.session_state.is_running = False
    st.session_state.inject_target_step = -1

def update_charts(data_placeholder, score_placeholder, metric_placeholder, result_label, result_color, current_val, pred_val, score, threshold, is_anomaly):
    plot_len = 100
    history = st.session_state.history
    current_step = history['step']
    
    # Lấy dữ liệu 100 điểm cuối
    display_data = history['data'][-plot_len:]
    display_preds = history['preds'][-plot_len:]
    display_scores = history['scores'][-plot_len:]
    display_thresholds = history['thresholds'][-plot_len:]
    display_anoms = history['anomalies'][-plot_len:]
    display_true = history['true_labels'][-plot_len:]
    
    # Tính toán x-axis thực tế
    x_range = range(max(0, current_step - len(display_data) + 1), current_step + 1)
    
    with data_placeholder.container():
        fig_data, ax_data = plt.subplots(figsize=(10, 3.5))
        ax_data.plot(x_range, display_data, color='cyan', label='Dữ liệu thực tế', alpha=0.8)
        ax_data.plot(x_range, display_preds, color='magenta', linestyle='--', label='Dự báo AI', alpha=0.8)
        
        # Ground Truth
        true_indices = [x_range[j] for j, val in enumerate(display_true) if val == 1]
        if true_indices:
            ax_data.scatter(true_indices, [display_data[j] for j, val in enumerate(display_true) if val == 1], 
                             color='red', marker='^', label='NGUY HIỂM (Thực tế)', s=100, zorder=5, edgecolors='white')
        
        # AI Predictions
        anom_indices = [x_range[j] for j, val in enumerate(display_anoms) if val == 1]
        if anom_indices:
            ax_data.scatter(anom_indices, [display_data[j] for j, val in enumerate(display_anoms) if val == 1], 
                             color='brown', marker='x', s=80, label='AI Cảnh báo', zorder=6)
        
        ax_data.set_title("📡 Đối chiếu AI vs Thực tế")
        ax_data.set_ylim(-0.1, 1.5)
        ax_data.legend(loc='upper right', fontsize='small')
        st.pyplot(fig_data)

    with score_placeholder.container():
        fig_score, ax_score = plt.subplots(figsize=(10, 2.5))
        ax_score.plot(x_range, display_scores, color='yellow', label='Sai số (MAE)')
        ax_score.plot(x_range, display_thresholds, color='orange', linestyle='--', label='Ngưỡng động')
        
        # Fill vùng bất thường
        scores_arr = np.array(display_scores)
        thresh_arr = np.array(display_thresholds)
        ax_score.fill_between(x_range, thresh_arr, scores_arr, where=(scores_arr > thresh_arr), color='red', alpha=0.3)
        
        ax_score.legend(loc='upper right', fontsize='small')
        st.pyplot(fig_score)
        
    with metric_placeholder.container():
        if result_color == "success": st.success(result_label)
        elif result_color == "warning": st.warning(result_label)
        elif result_color == "error": st.error(result_label)
        else: st.info(result_label)
        
        c1, c2 = st.columns(2)
        c1.metric("Lưu lượng", f"{current_val:.4f}")
        c2.metric("Sai số MAE", f"{score:.4f}", delta=f"{score-threshold:.4f}" if is_anomaly else None, delta_color="inverse")

# --- Sidebar (Top) ---
with st.sidebar:
    st.title("🛡️ Live Control")
    st.header("📂 Dữ liệu")
    sample_files = sorted([f for f in os.listdir('data') if f.endswith('.csv')])
    selected_file_name = st.selectbox("Chọn file mẫu:", sample_files)

# --- Load Data & Model ---
st.title("🛡️ Real-time Anomaly Detection Demo")
st.markdown("Mô phỏng việc giám sát luồng dữ liệu và phát hiện bất thường tức thời.")

model = load_anomaly_model()
if model is None:
    st.error("❌ Không tải được mô hình tại `checkpoint/seq2seq_anomaly_model.keras`.")
    st.stop()

df = pd.read_csv(os.path.join('data', selected_file_name))
scaler = MinMaxScaler()
full_data_scaled = scaler.fit_transform(df[['value']])
full_labels = df['is_anomaly'].values
LOOKBACK = 100
FORECAST_HORIZON = 10
ROLLING_WINDOW = 100
SHIFT_STEPS = 50

# Calculate Global Fixed STD ONCE per file to avoid recalculating
if 'global_fixed_std' not in st.session_state or st.session_state.get('current_file') != selected_file_name:
    st.session_state.current_file = selected_file_name
    fw = np.array([full_data_scaled[i-LOOKBACK:i].reshape(LOOKBACK, 1) for i in range(LOOKBACK, LOOKBACK+ROLLING_WINDOW)])
    fp = model.predict(fw, verbose=0)
    fs = [np.abs(full_data_scaled[i][0] - fp[idx, 0, 0]) for idx, i in enumerate(range(LOOKBACK, LOOKBACK+ROLLING_WINDOW))]
    st.session_state.global_fixed_std = np.std(fs) if np.std(fs) > 0 else 1e-4

# --- Sidebar (Bottom) ---
with st.sidebar:
    st.markdown("---")
    st.header("⚙️ Tham số")
    SPEED = st.slider("Tốc độ (s/step)", 0.001, 1.0, 0.1, step=0.001)
    K_STD = st.slider("Độ nhạy (K_STD)", 0.1, 6.0, 2.0)
    
    st.markdown("---")
    st.header("⏩ Tua nhanh")
    max_step = len(df) - FORECAST_HORIZON - 1
    jump_step = st.number_input(f"Nhảy đến bước (Max: {max_step}):", min_value=100, max_value=max_step, value=st.session_state.history['step'] if st.session_state.history['step'] >= 100 else 100)
    
    if jump_step != st.session_state.history['step']:
        st.session_state.history['step'] = jump_step
        st.toast(f"Đang tính toán bù dữ liệu quá khứ cho bước {jump_step}...")
        
        past_steps = ROLLING_WINDOW + SHIFT_STEPS
        start_calc = max(LOOKBACK, jump_step - past_steps)
        
        new_data, new_preds, new_scores, new_true, new_thresh, new_anom = [], [], [], [], [], []
        
        if start_calc < jump_step:
            windows = np.array([full_data_scaled[i-LOOKBACK:i].reshape(LOOKBACK, 1) for i in range(start_calc, jump_step)])
            batch_preds = model.predict(windows, verbose=0)
            
            for idx, i in enumerate(range(start_calc, jump_step)):
                curr_v = full_data_scaled[i][0]
                p = batch_preds[idx, 0, 0]
                s = np.abs(curr_v - p)
                new_data.append(curr_v)
                new_preds.append(p)
                new_scores.append(s)
                new_true.append(full_labels[i])
                
        for i in range(len(new_scores)):
            hist_sc = pd.Series(new_scores[:i+1])
            if len(hist_sc) > past_steps:
                rm = hist_sc.shift(SHIFT_STEPS).tail(ROLLING_WINDOW).mean()
            else:
                rm = hist_sc.mean()
            thresh = rm + (K_STD * st.session_state.global_fixed_std) + 1e-4
            new_thresh.append(thresh)
            new_anom.append(1 if new_scores[i] > thresh else 0)
            
        st.session_state.history['data'] = new_data
        st.session_state.history['preds'] = new_preds
        st.session_state.history['scores'] = new_scores
        st.session_state.history['thresholds'] = new_thresh
        st.session_state.history['anomalies'] = new_anom
        st.session_state.history['true_labels'] = new_true
        st.toast(f"Đã nhảy đến bước {jump_step} và điền đầy đủ lịch sử!")
    
    st.markdown("---")
    if st.button("🔄 Reset Demo"):
        reset_simulation()
        st.rerun()

# UI Layout
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("🛠️ Inject Anomalies")
    if st.button("💥 Chèn Lỗi sau 10 bước", use_container_width=True):
        st.session_state.inject_target_step = st.session_state.history['step'] + 10
        st.warning(f"Sẽ chèn lỗi tại bước {st.session_state.inject_target_step}")
    
    st.markdown("---")
    if not st.session_state.is_running:
        btn_label = "▶️ Bắt đầu Chạy" if st.session_state.history['step'] == 0 else "▶️ Tiếp tục"
        if st.button(btn_label, type="primary", use_container_width=True):
            st.session_state.is_running = True
            st.rerun()
    else:
        if st.button("⏸️ Dừng", use_container_width=True):
            st.session_state.is_running = False
            st.rerun()

    # Metrics Section
    st.subheader("📊 Trạng thái")
    m_placeholder = st.empty()

with col1:
    chart_data_placeholder = st.empty()
    chart_score_placeholder = st.empty()

# --- Vòng lặp Simulation ---
if st.session_state.is_running:
    start_idx = max(LOOKBACK, st.session_state.history['step'])
    
    for i in range(start_idx, len(df) - FORECAST_HORIZON):
        if not st.session_state.is_running:
            break
            
        current_val = full_data_scaled[i][0]
        true_label = full_labels[i]
        
        # Injection
        target_step = st.session_state.get('inject_target_step', -1)
        if i == target_step:
            current_val += np.random.uniform(0.6, 1.0)
            st.session_state.inject_target_step = -1
        elif target_step != -1 and i < target_step:
            st.sidebar.write(f"⏳ Lỗi tại bước {target_step} (còn {target_step - i})")
            
        # 1. Dự đoán
        window = full_data_scaled[i-LOOKBACK:i].reshape(1, LOOKBACK, 1)
        
        pred = model.predict(window, verbose=0)
        # Tính sai số (MAE) tuyệt đối chỉ dựa trên điểm thực tại hiện tại (Chuẩn Real-time, không rò rỉ dữ liệu tương lai)
        score = np.abs(current_val - pred[0, 0, 0])
        
        # 2. Cập nhật Lịch sử
        st.session_state.history['data'].append(current_val)
        st.session_state.history['preds'].append(pred[0, 0, 0])
        st.session_state.history['scores'].append(score)
        st.session_state.history['true_labels'].append(true_label)
        st.session_state.history['step'] = i
        
        # Adaptive Threshold (Đồng bộ với Notebook)
        ROLLING_WINDOW = 100
        SHIFT_STEPS = 50
        
        history_scores = pd.Series(st.session_state.history['scores'])
        
        # Tính toán rolling mean động với độ trễ (Shift)
        if len(history_scores) > (ROLLING_WINDOW + SHIFT_STEPS):
            # Lấy cửa sổ dữ liệu quá khứ cách đây 50 bước
            window_scores = history_scores.shift(SHIFT_STEPS).tail(ROLLING_WINDOW)
            rolling_mean = window_scores.mean()
        else:
            rolling_mean = history_scores.mean()
            
        # Dùng global_fixed_std đã tính ở đầu file
        fixed_std = st.session_state.global_fixed_std
            
        epsilon = 1e-4
        
        threshold = rolling_mean + (K_STD * fixed_std) + epsilon
        st.session_state.history['thresholds'].append(threshold)
        
        # 3. Phân loại
        is_anomaly = 1 if score > threshold else 0
        if is_anomaly == 1 and true_label == 1: result_label, result_color = "🎯 ĐÚNG (True Positive)", "success"
        elif is_anomaly == 1 and true_label == 0: result_label, result_color = "⚠️ SAI (False Positive)", "warning"
        elif is_anomaly == 0 and true_label == 1: result_label, result_color = "🚫 BỎ SÓT (False Negative)", "error"
        else: result_label, result_color = "✅ BÌNH THƯỜNG", "light"
        
        st.session_state.history['anomalies'].append(is_anomaly)
        
        # 4. Vẽ Chart
        update_charts(chart_data_placeholder, chart_score_placeholder, m_placeholder, result_label, result_color, current_val, pred[0, 0, 0], score, threshold, is_anomaly)
        
        time.sleep(SPEED)
else:
    # Nếu đang dừng nhưng đã có dữ liệu trong lịch sử, hãy hiển thị lại đồ thị cuối cùng
    if st.session_state.history['step'] > 0 and len(st.session_state.history['data']) > 0:
        history = st.session_state.history
        # Lấy các giá trị cuối cùng để hiển thị metrics
        last_val = history['data'][-1]
        last_pred = history['preds'][-1]
        last_score = history['scores'][-1]
        last_thresh = history['thresholds'][-1]
        last_anom = history['anomalies'][-1]
        last_true = history['true_labels'][-1]
        
        if last_anom == 1 and last_true == 1: res_l, res_c = "🎯 ĐÚNG (True Positive)", "success"
        elif last_anom == 1 and last_true == 0: res_l, res_c = "⚠️ SAI (False Positive)", "warning"
        elif last_anom == 0 and last_true == 1: res_l, res_c = "🚫 BỎ SÓT (False Negative)", "error"
        else: res_l, res_c = "✅ BÌNH THƯỜNG", "light"
        
        update_charts(chart_data_placeholder, chart_score_placeholder, m_placeholder, res_l, res_c, last_val, last_pred, last_score, last_thresh, last_anom)
    else:
        st.info("Nhấn 'Bắt đầu Chạy' để mô phỏng luồng dữ liệu.")
