# So sánh Autoencoder, Random Forest, Isolation Forest và One-Class SVM

## Chạy trên Windows

Mở Terminal PowerShell hoặc Git Bash tại thư mục project:

```powershell
./.venv/Scripts/python.exe scripts/compare_models.py
```

Lệnh này huấn luyện ba baseline, chọn threshold bằng validation và đánh giá
cả bốn mô hình trên KDDTest+. Autoencoder tối ưu được nạp lại, không huấn luyện
lại. One-Class SVM học trên toàn bộ Normal train nên có thể mất vài phút.
Chạy lại sẽ ghi đè baseline và báo cáo so sánh của lần trước.

Cần có dữ liệu processed và Autoencoder tối ưu từ các giai đoạn trước. Nếu chưa có:

```powershell
./.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
./.venv/Scripts/python.exe scripts/prepare_nsl_kdd.py
./.venv/Scripts/python.exe scripts/optimize_autoencoder.py
./.venv/Scripts/python.exe scripts/compare_models.py
```

## Giao thức

- Dùng cùng 100.778 mẫu train, 25.195 mẫu validation và 22.544 mẫu KDDTest+.
- Cùng 77 đặc trưng từ pipeline mã hóa/StandardScaler fit trên Normal train.
- Random Forest dùng toàn bộ phần train có nhãn; AE/IF/OCSVM chỉ học từ
  53.874 mẫu Normal. RF là baseline có giám sát và phải được diễn giải riêng.
- RF: 300 cây, class_weight=balanced, seed 42, n_jobs=4.
- IF: 300 cây, max_samples=256 mỗi cây, random_state=42, n_jobs=4. Gọi fit trên
  toàn bộ Normal train; lấy mẫu mỗi cây là cơ chế thuật toán, không phải bỏ bớt
  dữ liệu trước khi huấn luyện.
- OCSVM: RBF, gamma=scale, nu=0.05, cache 1024 MB, max_iter=-1. Kiểm tra hội tụ;
  không chấp nhận fit_status khác 0.
- Cố định cấu hình baseline trước khi tính test. Đây chưa phải tìm kiếm siêu tham
  số toàn diện; AE có ngân sách tối ưu khác do đã thử ba latent dimension.
- Nạp nguyên checkpoint, threshold và pipeline của Autoencoder tối ưu. Xác minh
  hash dữ liệu và pipeline khớp với lần tối ưu trước; thay đổi dữ liệu cần chạy lại
  tối ưu để có đối chiếu hợp lệ.
- Baseline dùng đúng 4.040 Normal calibration và 7.559 mẫu selection của AE.
- Chọn trong P97/P98/P99/P99.5/P99.9 Normal calibration: macro-F1 selection cao
  nhất với FPR<=1%. Nếu không có ứng viên đạt thì chọn FPR thấp nhất và báo rõ.
  Khi trùng điểm ở threshold vẫn dùng quy tắc `score >= threshold` là Attack.
- RF dùng xác suất lớp Attack; IF dùng `-score_samples`; OCSVM dùng
  `-decision_function`. Score lớn hơn luôn là đáng ngờ hơn. Giá trị score giữa
  các mô hình không cùng đơn vị; score OCSVM có thể âm.
- Có dùng nhãn validation để chọn threshold. Không dùng nhãn test để chọn
  mô hình, ngưỡng hoặc preprocessing. Test đã được xem ở những giai đoạn trước;
  lần chạy này không phải một tập holdout chưa từng quan sát trong cả dự án.

## Đầu ra

- [Bảng so sánh và phân tích](../reports/model_comparison/comparison.md)
- [Bảng CSV bốn mô hình](../reports/model_comparison/comparison.csv)
- [Metrics và cấu hình đầy đủ](../reports/model_comparison/metrics.json)
- [Recall/FPR theo nhóm](../reports/model_comparison/group_detection.csv)
- [Đường PR/ROC](../reports/model_comparison/comparison_curves.png)

Mỗi mô hình có file `*_predictions.csv` với cùng row_id, nhãn thật, nhóm,
score, threshold và dự đoán Normal/Attack. Confusion matrix có hàng là nhãn thật,
cột là dự đoán, thứ tự Normal/Attack. PR-AUC tính theo average precision.

Baseline lưu trong `artifacts/models/baselines/`: ba file joblib gồm estimator,
threshold, schema và cấu hình; kèm pipeline.joblib, validation_partition.npz,
metadata.json. Chỉ nạp joblib từ nguồn tin cậy và giữ cùng phiên bản thư viện.
Autoencoder vẫn nằm trong `artifacts/models/autoencoder_optimized/`.

Kiểm thử:

```powershell
./.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_*.py' -v
./.venv/Scripts/python.exe scripts/verify_model_comparison.py
```

Code: [compare_models.py](../scripts/compare_models.py),
[baselines.py](../src/nsl_kdd/baselines.py).
