# EDA và tiền xử lý

## Chạy lại

Môi trường hiện tại dùng Python 3.14.5. Từ thư mục gốc, trong PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe scripts\run_preparation_notebook.py
.\.venv\Scripts\python.exe -m jupyterlab notebooks\01_eda_preprocessing.ipynb
```

Lần thiết lập đầu có thể cài từ `requirements.txt`; `requirements-lock.txt` ghi
phiên bản đã chạy thành công trên Windows/Python 3.14. Không cần kích hoạt venv.
Khi mở bằng VS Code, chọn interpreter `.venv/Scripts/python.exe` cho notebook.

Notebook [01_eda_preprocessing.ipynb](../notebooks/01_eda_preprocessing.ipynb)
chứa EDA, biểu đồ, kiểm tra chia tập và kiểm tra lưu/nạp artifact. Chạy độc lập
preprocessing bằng `python scripts/prepare_nsl_kdd.py` trong môi trường đã cài.

## Giao thức dữ liệu

- Raw train/test được xác minh theo hash của mirror đã pin; chỉ tải khi thiếu.
- Validation 20%, seed 42, stratify nhãn nhị phân; test giữ nguyên.
- Chỉ normal của phần train được dùng để fit StandardScaler và OneHotEncoder.
- Không đưa label, difficulty, group hoặc is_attack vào đặc trưng.
- Category chưa thấy có block one-hot bằng 0; notebook thống kê theo split.
- StandardScaler không clip; giá trị có thể âm hoặc rất lớn ngoài train.
  Khi huấn luyện AE cần đầu ra linear và xem xét ảnh hưởng outlier đến MSE.
- Không tự xóa bản ghi trùng hoặc cột hằng; giữ giao thức benchmark.

## Artifact

- `data/processed/nsl_kdd.npz`: X_train, X_validation, X_test và các y/group/label/row
  tương ứng; X_train_normal, X_validation_normal cho AE.
- `artifacts/preprocessing/pipeline.joblib`: pipeline sklearn đã fit, nạp bằng joblib.load.
- `artifacts/preprocessing/metadata.json`: seed, chính sách, tên đặc trưng,
  kích thước và phiên bản thư viện.

Các file sinh ra được bỏ qua bởi Git và tái tạo được từ notebook. NPZ đọc bằng
`np.load(path, allow_pickle=False)`. Chỉ nạp joblib đáng tin cậy, dùng cùng phiên bản
thư viện. Row ID bắt đầu từ 0 trong file raw riêng của mỗi tập.

Chưa huấn luyện Autoencoder và chưa có điểm rủi ro ở giai đoạn này. Test chỉ được
khảo sát mô tả, không được dùng chọn cấu hình hay ngưỡng.
