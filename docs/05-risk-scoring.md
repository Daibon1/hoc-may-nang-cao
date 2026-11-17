# Risk scoring và tối ưu Autoencoder

## Chạy

Mở PowerShell hoặc Git Bash tại thư mục project:

```powershell
./.venv/Scripts/python.exe scripts/optimize_autoencoder.py
./.venv/Scripts/python.exe scripts/predict_autoencoder.py
```

Lệnh đầu huấn luyện lại ba cấu hình và ghi đè kết quả tối ưu của lần trước;
cần vài phút trên CPU. Lệnh thứ hai chỉ nạp mô hình đã lưu để suy luận.
Không cần activate venv. Cài môi trường theo `requirements-lock.txt` nếu chuyển máy.

CSV suy luận: `reports/risk_optimization/predictions.csv`, gồm `row_id`,
`reconstruction_error`, `threshold`, `prediction`, `risk_score`, `alert_level`.
Đổi dữ liệu hoặc nơi lưu bằng `--input` / `--output`. File vào không header,
có 41 đặc trưng hoặc 43 cột NSL-KDD gốc; không cần nhãn để dự đoán.

## Công thức điểm và ba mức cảnh báo

Với lỗi tái tạo mỗi dòng là MSE trên toàn bộ đặc trưng, điểm được định nghĩa:

```text
risk_score(error) = 100 × số mẫu Normal calibration có MSE <= error / số mẫu tham chiếu
```

Đây là phân vị thực nghiệm của mức bất thường, không phải xác suất tấn công hoặc
mức thiệt hại. Điểm nằm trong [0,100], không giảm khi MSE tăng; các giá trị bằng
nhau nhận cùng điểm. Ngoài miền tham chiếu, điểm bão hòa ở 0 hoặc 100, nên giữ
reconstruction error để phân biệt các mẫu có cùng điểm 100.

| Mức | Quy tắc theo MSE | Hành động diễn giải |
|---|---|---|
| Low | Dưới P95 Normal calibration | Theo dõi thông thường |
| Medium | Từ P95 đến dưới ngưỡng High đã chọn | Cảnh báo sớm, cần xem xét |
| High | Từ ngưỡng High trở lên | Cảnh báo chính; dự đoán Attack |

Low/Medium được gán Normal trong bài toán nhị phân. Không làm tròn MSE trước khi
so sánh. Ngưỡng mức cảnh báo dùng MSE gốc: do phép nội suy quantile, ties và làm
tròn, điểm hiển thị sát ranh giới có thể không trùng chính xác mốc 95/99.
Mức và prediction xuất bởi chương trình là kết quả quyết định chính thức.

## Giao thức so sánh

- Giữ nguyên split train/validation/test của preprocessing; chỉ học trọng số từ Normal train.
- Chia validation 40%/30%/30%, stratify nhãn nhị phân, seed 42. Phần 40% chỉ dùng
  Normal cho early stopping; phần 30% thứ nhất chỉ dùng Normal cho calibration;
  phần 30% cuối giữ cả Normal/Attack để chọn latent dimension và threshold.
- Ba kiến trúc có latent 8, 16, 32, cùng encoder/decoder hidden 64/32, optimizer,
  seed, batch size, weight decay, giới hạn epoch và patience. Huấn luyện tất cả
  từ đầu theo cùng giao thức, không tái dùng mô hình của giai đoạn trước.
- Chọn latent có PR-AUC (average precision từ MSE) lớn nhất trên validation selection;
  latent nhỏ hơn phá hòa khi bằng nhau. Không chọn bằng MSE train thấp nhất.
- Thử threshold P95/P97/P98/P99/P99.5/P99.9 tính từ Normal calibration.
  P95 dành cho mức Medium. Trong các ngưỡng lớn hơn P95, chọn macro-F1 cao nhất
  với FPR <=1% trên validation selection; hòa chọn ngưỡng cao hơn. Nếu không có
  ngưỡng đạt ngân sách thì chọn FPR thấp nhất và báo rõ constraint không đạt.
- Sau khi chốt model, reference, ngưỡng Low/Medium/High mới đọc test để đánh giá
  mô hình thắng. Không chạy test cho cả ba rồi chọn cấu hình tốt nhất.
- Chính sách chọn threshold có sử dụng nhãn validation; đây là calibration có
  giám sát trên validation, còn huấn luyện Autoencoder vẫn chỉ dùng Normal.

## Phân tích overfitting và giới hạn

So sánh loss train và early-stopping validation theo từng epoch; khôi phục
checkpoint có loss Normal validation thấp nhất. Báo cáo ghi epoch tốt nhất,
số epoch thực chạy, MSE train của checkpoint đó, MSE validation và khoảng chênh.
Loss train trong đường cong là trung bình online trong epoch; MSE train cuối
trong bảng là đánh giá lại checkpoint đã chọn, vì vậy chúng có thể khác nhau.

Loss validation tăng trong khi loss train giảm là dấu hiệu cần theo dõi; một vài
dao động không đủ khẳng định overfitting do dữ liệu có đuôi dài và gradient ngẫu
nhiên. Early stopping và weight decay giúp hạn chế, không chứng minh đã loại bỏ
overfitting. Một seed cho mỗi latent chỉ là khảo sát ban đầu, chưa có độ tin cậy
qua nhiều seed. AP cao không đảm bảo recall cao tại ngưỡng FPR thấp.

Test NSL-KDD đã được xem ở giai đoạn trước; lần tìm kiếm này không dùng kết quả
test làm tiêu chí chọn. Không xem kết quả benchmark này là bằng chứng hiệu quả
trên lưu lượng mạng hiện đại.

## File đầu ra

- `artifacts/models/autoencoder_optimized/`: model.pt, pipeline.joblib,
  normal_reference.npy, risk_config.json, config.json và validation_partition.npz.
  Cần giữ nguyên cả thư mục để suy luận trên máy khác.
- `artifacts/models/latent_trials/`: checkpoint cho từng latent dimension.
- `reports/risk_optimization/`: summary.json, latent_comparison.csv,
  threshold_analysis.csv, history_latent_*.csv, risk_by_group.csv,
  test_risk_predictions.csv và optimization.png.

Kết quả số liệu và diễn giải: [Báo cáo tối ưu](../reports/risk_optimization/analysis.md).
Lệnh dự đoán mặc định dùng model tối ưu. Muốn dùng model P99 giai đoạn trước:

```powershell
./.venv/Scripts/python.exe scripts/predict_autoencoder.py --model-dir artifacts/models/autoencoder --output reports/autoencoder/predictions.csv
```

Code: [risk.py](../src/nsl_kdd/risk.py), [training.py](../src/nsl_kdd/training.py),
[optimize_autoencoder.py](../scripts/optimize_autoencoder.py).
