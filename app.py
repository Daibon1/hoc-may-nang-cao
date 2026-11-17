"""Streamlit demo: upload raw NSL-KDD CSV and inspect calibrated anomaly risk."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from nsl_kdd.dashboard import parse_csv, load_bundle, score_frame, FEATURES

st.set_page_config(page_title='Network Risk · NSL-KDD', page_icon='🛡️', layout='wide')


@st.cache_resource
def detector():
    torch.set_num_threads(1)
    return load_bundle(ROOT / 'artifacts/models/autoencoder_optimized')


@st.cache_data
def sample_csv():
    frame = pd.read_csv(ROOT / 'data/raw/KDDTest+.txt', header=None, nrows=100)
    frame.columns = [*FEATURES, 'label', 'difficulty']
    return frame.to_csv(index=False).encode('utf-8-sig')


with st.sidebar:
    st.title('🛡️ Network Risk')
    st.caption('NSL-KDD · Autoencoder')
    st.divider()
    st.markdown('**Ba mức cảnh báo**')
    st.markdown('🟢 **Low** · Theo dõi thông thường\n\n🟠 **Medium** · Cần xem xét\n\n🔴 **High** · Cảnh báo chính / Attack')
    st.divider()
    st.caption('Điểm 0–100 là phân vị bất thường so với Normal calibration, không phải xác suất tấn công.')
    st.caption('Dữ liệu chỉ được xử lý trong bộ nhớ phiên; ứng dụng không ghi file upload lên ổ đĩa.')

st.title('Phân tích rủi ro lưu lượng mạng')
st.write('Tải dữ liệu NSL-KDD lên để xem điểm bất thường, phát hiện mẫu đáng chú ý và xuất kết quả.')
try:
    bundle = detector()
except (OSError, ValueError, RuntimeError, KeyError) as exc:
    st.error(f'Không nạp được mô hình tối ưu: {exc}')
    st.code('./.venv/Scripts/python.exe scripts/optimize_autoencoder.py', language='bash')
    st.stop()

with st.expander('Định dạng CSV được hỗ trợ'):
    st.markdown('**Dữ liệu gốc, chưa scaling**, UTF-8, phân cách dấu phẩy, tối đa **20 MB / 100.000 dòng**. '
                'Có header: đủ 41 tên đặc trưng; tùy chọn `label`, `difficulty`, thứ tự cột bất kỳ. '
                'Không header: đúng 41 hoặc 43 cột theo thứ tự NSL-KDD. Nhãn và difficulty không dùng để dự đoán.')
    st.code(','.join(FEATURES), language='text')
    st.download_button('Tải CSV mẫu (100 dòng)', sample_csv(), 'nsl_kdd_sample.csv', 'text/csv')
    st.caption('CSV mẫu lấy từ KDDTest+ để demo, không phải dữ liệu mới để huấn luyện.')

left, right = st.columns([3, 1])
with left:
    uploaded = st.file_uploader('Chọn CSV lưu lượng mạng', type=['csv', 'txt'], max_upload_size=20)
with right:
    mode_label = st.selectbox('Header', ['Tự nhận diện', 'Có header', 'Không header'])
    demo = st.checkbox('Dùng 100 dòng mẫu', value=False)

content = uploaded.getvalue() if uploaded is not None else (sample_csv() if demo else None)
if content is None:
    st.info('Upload CSV hoặc chọn “Dùng 100 dòng mẫu” để bắt đầu.')
    st.stop()

try:
    frame, format_label = parse_csv(content, {'Tự nhận diện': 'auto', 'Có header': 'header', 'Không header': 'no_header'}[mode_label])
except ValueError as exc:
    st.error(f'Kiểm tra schema thất bại: {exc}')
    st.stop()
st.success(f'Schema hợp lệ · {len(frame):,} dòng · {format_label} · 41 đặc trưng đầu vào')

# Keep computed results in this session, never in a global upload cache.
import hashlib
identity = hashlib.sha256(content + mode_label.encode()).hexdigest()
if st.session_state.get('input_identity') != identity:
    st.session_state.pop('scoring_result', None)
    st.session_state['input_identity'] = identity

if st.button('Phân tích rủi ro', type='primary'):
    try:
        with st.spinner('Đang tính reconstruction error và risk score…'):
            st.session_state['scoring_result'] = score_frame(frame, bundle)
    except (ValueError, RuntimeError) as exc:
        st.error(f'Không thể dự đoán: {exc}')
        st.stop()
if 'scoring_result' not in st.session_state:
    st.caption('Dữ liệu đã sẵn sàng. Nhấn “Phân tích rủi ro” để chạy mô hình.')
    st.stop()

result, unknown = st.session_state['scoring_result']
if any(unknown.values()):
    st.warning('Có category chưa thấy trong Normal train (mã hóa bằng block 0): ' +
               ', '.join(f'{name}: {count} dòng' for name, count in unknown.items() if count))

counts = result.alert_level.value_counts()
columns = st.columns(4)
columns[0].metric('Tổng số mẫu', f'{len(result):,}')
columns[1].metric('🔴 High / Attack', f'{counts.get("High", 0):,}')
columns[2].metric('🟠 Medium', f'{counts.get("Medium", 0):,}')
columns[3].metric('🟢 Low', f'{counts.get("Low", 0):,}')
st.caption(f'Tỷ lệ cảnh báo High: {result.prediction.eq("Attack").mean():.2%}. '
           'Đây là tỷ lệ dự đoán; không phải tỷ lệ tấn công thật hoặc độ chính xác.')

palette = alt.Scale(domain=['Low', 'Medium', 'High'], range=['#14967c', '#e6a23c', '#e05260'])
chart_left, chart_right = st.columns(2)
with chart_left:
    st.subheader('Phân bố Risk Score')
    edges = np.linspace(0, 100, 21)
    hist, _ = np.histogram(result.risk_score, bins=edges)
    bins = pd.DataFrame({'score': edges[:-1], 'end': edges[1:], 'count': hist})
    st.altair_chart(alt.Chart(bins).mark_bar(color='#279caa').encode(
        x=alt.X('score:Q', bin='binned', title='Risk Score (0–100)', scale=alt.Scale(domain=[0, 100])),
        x2='end:Q', y=alt.Y('count:Q', title='Số mẫu'), tooltip=['score', 'end', 'count']), width='stretch')
with chart_right:
    st.subheader('Mức cảnh báo')
    levels = pd.DataFrame({'level': ['Low', 'Medium', 'High'],
                           'count': [counts.get(level, 0) for level in ['Low', 'Medium', 'High']]})
    st.altair_chart(alt.Chart(levels).mark_bar().encode(
        x=alt.X('level:N', sort=['Low', 'Medium', 'High'], title='Mức cảnh báo'),
        y=alt.Y('count:Q', title='Số mẫu'), color=alt.Color('level:N', scale=palette, legend=None),
        tooltip=['level', 'count']), width='stretch')

st.subheader('Kết quả từng mẫu')
filters, export = st.columns([2, 1])
with filters:
    selected = st.multiselect('Lọc mức cảnh báo', ['High', 'Medium', 'Low'], default=['High', 'Medium', 'Low'])
with export:
    st.download_button('⬇ Tải toàn bộ kết quả CSV', result.to_csv(index=False).encode('utf-8-sig'),
                       'network_risk_predictions.csv', 'text/csv', type='primary')
view = result[result.alert_level.isin(selected)].sort_values('risk_score', ascending=False, kind='stable')
st.caption(f'{len(view):,} dòng sau lọc. Hiển thị tối đa 1.000 dòng; file tải chứa toàn bộ {len(result):,} dòng theo thứ tự gốc. '
           'row_id bắt đầu từ 0 trong dữ liệu upload.')
st.dataframe(view.head(1000), hide_index=True, width='stretch', column_config={
    'risk_score': st.column_config.ProgressColumn('Risk Score', min_value=0, max_value=100, format='%.2f'),
    'reconstruction_error': st.column_config.NumberColumn('Reconstruction MSE', format='%.8f')})
with st.expander('Ngưỡng vận hành và giới hạn'):
    config = bundle[2]
    st.write(f'Medium: MSE ≥ {config["medium_threshold"]:.8f}; High: MSE ≥ {config["high_threshold"]:.8f}. '
             'High được dự đoán Attack; Low/Medium được dự đoán Normal. Ngưỡng đã chốt bằng validation.')
    st.write('Không tự thay đổi ngưỡng theo file upload. Điểm hiển thị được làm tròn; mức cảnh báo dùng MSE gốc. '
             'NSL-KDD là benchmark cũ; kết quả demo không chứng minh hiệu quả trên lưu lượng mạng hiện đại.')
