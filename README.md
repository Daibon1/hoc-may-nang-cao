# NSL-KDD Anomaly Risk Scoring

## Bộ nộp và thuyết trình

[Báo cáo, PowerPoint, kịch bản demo và ZIP dự án](deliverables/README.md).
Kiểm thử tích hợp: `scripts/audit_submission.py`; tạo tài liệu: `scripts/build_submission.py`.

## Streamlit dashboard

```powershell
./.venv/Scripts/python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

Mở http://127.0.0.1:8501 hoặc nhấp đúp [run_demo.cmd](run_demo.cmd).
Upload CSV, kiểm tra schema, xem Risk Score/Low–Medium–High, biểu đồ và tải kết quả.
[Hướng dẫn demo và schema upload](docs/07-streamlit-demo.md).

## Bảng so sánh bốn mô hình

```powershell
./.venv/Scripts/python.exe scripts/compare_models.py
```

[Bảng kết quả](reports/model_comparison/comparison.md) ·
[Hướng dẫn và giao thức](docs/06-model-comparison.md).

RF học có giám sát từ train; Isolation Forest, One-Class SVM và Autoencoder học
từ Normal. Cùng preprocessing, validation và KDDTest+; ngưỡng được chọn từ validation.

## Risk score 0–100 và ba mức cảnh báo

[Hướng dẫn tối ưu latent 8/16/32 và risk scoring](docs/05-risk-scoring.md)

```powershell
./.venv/Scripts/python.exe scripts/optimize_autoencoder.py
./.venv/Scripts/python.exe scripts/predict_autoencoder.py
```

Lệnh dự đoán mặc định dùng mô hình tối ưu và xuất `risk_score`, `alert_level`
(Low/Medium/High), Normal/Attack vào `reports/risk_optimization/predictions.csv`.
Điểm là phân vị lỗi tái tạo theo Normal calibration, không phải xác suất tấn công.

## Autoencoder Normal/Attack

[Hướng dẫn huấn luyện và dự đoán](docs/04-autoencoder.md)

```powershell
./.venv/Scripts/python.exe scripts/train_autoencoder.py
./.venv/Scripts/python.exe scripts/predict_autoencoder.py --model-dir artifacts/models/autoencoder --output reports/autoencoder/predictions.csv
```

Mô hình học từ Normal train, dừng sớm bằng Normal validation và chọn ngưỡng P99
trên phần Normal validation dành riêng cho calibration. Kết quả lưu trong
`reports/autoencoder/`; mô hình và preprocessing đi kèm nằm ở `artifacts/models/autoencoder/`.

## EDA và preprocessing đã triển khai

- [Notebook EDA và preprocessing](notebooks/01_eda_preprocessing.ipynb)
- [Hướng dẫn môi trường, chạy lại và artifact](docs/03-eda-preprocessing.md)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\run_preparation_notebook.py
```

Pipeline chia validation 20% với seed 42, fit encoder/scaler chỉ trên normal của
phần train và lưu dữ liệu cùng pipeline để huấn luyện Autoencoder. Phần mô tả
cấu trúc và trạng thái khảo sát bên dưới được giữ làm bối cảnh giai đoạn trước;
hướng dẫn giai đoạn hiện tại nằm trong liên kết ở trên.

Project nghiên cứu phát hiện lưu lượng bất thường trên NSL-KDD bằng
Autoencoder, chấm điểm theo reconstruction error và so sánh với các baseline
machine learning.

## Cấu trúc hiện tại

```text
hoc-may-nc/
|-- data/
|   `-- raw/                 # NSL-KDD gốc; không chỉnh sửa
|-- docs/                    # Phạm vi và tài liệu theo giai đoạn
|-- reports/                 # Báo cáo khảo sát và kết quả thực nghiệm
|-- scripts/                 # Lệnh chạy độc lập, hiện có công cụ khảo sát dữ liệu
|-- tests/                   # Kiểm thử tự động
|-- .venv/                   # Môi trường cục bộ, không đưa vào Git
|-- .gitignore
`-- README.md
```

## Cấu trúc code hướng tới

Khi bắt đầu giai đoạn tiền xử lý và huấn luyện, đặt code có thể tái sử dụng
trong `src/nsl_kdd/`; giữ `scripts/` làm các điểm chạy pipeline, không chép
logic xử lý dữ liệu vào nhiều script.

```text
src/nsl_kdd/
|-- data/                    # Đọc dữ liệu, schema và ánh xạ nhãn
|-- preprocessing/           # Encode, scale và tạo train/validation
|-- models/                  # Autoencoder và các baseline
`-- evaluation/              # Metrics, ngưỡng và mức điểm rủi ro
```

Tạo module trong các thư mục này khi có phần triển khai tương ứng; không để
các package rỗng chỉ để giữ chỗ. Nếu có artifact được tạo, dùng các thư mục
`data/processed/`, `artifacts/models/` và `reports/experiments/` theo đúng mục
đích, không ghi đè dữ liệu gốc.

## Môi trường và cách chạy

Kích hoạt môi trường ảo trong PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Khảo sát dữ liệu và tạo lại báo cáo:

```powershell
python scripts\inspect_nsl_kdd.py
```

Nếu chưa có dữ liệu, thêm `--download`. Sau khi khảo sát, báo cáo được ghi vào
`reports\nsl_kdd_data_profile.json`. Kết quả tóm tắt được in trực tiếp trong
PowerShell; sơ đồ Mermaid trong tài liệu trình bày các số liệu đó mà không cần
tạo file ảnh.

Chạy kiểm thử hiện có:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

`inspect_nsl_kdd.py` hiện chỉ dùng thư viện chuẩn Python. Khai báo các thư viện
machine learning và deep learning trong manifest dependency khi giai đoạn
huấn luyện chọn và sử dụng chúng.

## Tiến độ tài liệu

- [Giai đoạn 1: phạm vi và tiêu chí](docs/01-pham-vi-va-tieu-chi.md)
- [Giai đoạn 2: chuẩn bị và khảo sát dữ liệu](docs/02-khao-sat-du-lieu.md)
