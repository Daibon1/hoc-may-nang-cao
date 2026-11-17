# Web demo Streamlit

## Mở ứng dụng

Trong PowerShell hoặc Git Bash, tại thư mục project:

```powershell
./.venv/Scripts/python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

Mở **http://127.0.0.1:8501**. Giữ terminal chạy; Ctrl+C để dừng server.
Trên Windows cũng có thể nhấp đúp [run_demo.cmd](../run_demo.cmd).
Không cần activate venv hoặc dùng lệnh `streamlit` từ PATH.
Nếu port 8501 đang được dùng, dùng server đang mở hoặc đổi `--server.port 8502`.

Nếu chưa cài môi trường:

```powershell
./.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
```

Ứng dụng cần bộ model tối ưu hiện có trong `artifacts/models/autoencoder_optimized/`:
model.pt, pipeline.joblib, normal_reference.npy và risk_config.json. Nếu thiếu,
chạy lại preprocessing/optimization theo các giai đoạn trước. Không huấn luyện
lại khi nhấn phân tích trên dashboard.

## Sử dụng

1. Chọn **Dùng 100 dòng mẫu** hoặc upload CSV/TXT.
2. Chọn cách nhận diện header nếu cần; ứng dụng hiển thị lỗi cụ thể nếu schema sai.
3. Nhấn **Phân tích rủi ro**.
4. Xem tổng số mẫu, High/Attack, Medium và Low; hai biểu đồ phân bố Risk Score
   và số mẫu mỗi mức cảnh báo.
5. Lọc mức cảnh báo trong bảng; điểm 0–100 có thanh biểu diễn và MSE gốc.
6. Nhấn **Tải toàn bộ kết quả CSV**. File chứa toàn bộ dòng theo thứ tự upload,
   không chỉ phần đang lọc hoặc 1.000 dòng xem trước.

## Schema upload

- CSV UTF-8 (có/không BOM), phân cách bằng dấu phẩy.
- Dữ liệu NSL-KDD **gốc, chưa scaling**, tối đa 20 MB / 100.000 mẫu.
- Có header: đủ 41 tên đặc trưng của NSL-KDD; thứ tự bất kỳ; tùy chọn `label`,
  `difficulty`. Tên cột phải đúng, không nhận cột thừa hoặc cột trùng.
- Không header: đúng 41 đặc trưng hoặc 43 cột gốc theo thứ tự NSL-KDD.
- Không nhận file rỗng, thiếu trường, sai số cột, giá trị trống, chuỗi ở cột số,
  NaN/Inf, số âm, số thập phân ở cột đếm, rate ngoài [0,1], cờ binary sai miền.
- Category chưa thấy khi fit được cảnh báo và mã hóa bằng block one-hot 0 theo
  pipeline đã lưu; không tự fit lại encoder hoặc scaler bằng dữ liệu upload.
- `label` và `difficulty` không tham gia suy luận. Nếu có label, giữ trong kết quả
  dưới tên `original_label` để tham khảo; không dùng nó để tự điều chỉnh ngưỡng.

Dashboard có nút tải CSV mẫu có header gồm 100 dòng đầu KDDTest+. Có thể kiểm tra
full dataset bằng cách upload `data/raw/KDDTest+.txt`.

## Diễn giải

Risk Score là phân vị lỗi tái tạo so với Normal calibration, không phải xác suất
tấn công hoặc mức thiệt hại. High tương ứng Attack; Low/Medium tương ứng Normal
trong quyết định nhị phân. Ngưỡng được chốt từ validation, không chỉnh trên upload.
Tỷ lệ High hiển thị là tỷ lệ dự đoán, không phải độ chính xác hay tỷ lệ nhãn thật.
Điểm hiển thị được làm tròn nhưng cảnh báo được tính từ MSE đầy đủ.

Upload và kết quả suy luận giữ trong bộ nhớ phiên, không ghi file upload lên ổ đĩa.
Model được cache để tải một lần; khởi động lại server sau khi thay bộ model.
Server demo mặc định chỉ lắng nghe localhost; chưa triển khai hosting công khai.

## Kiểm chứng

```powershell
./.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_*.py' -v
```

Kiểm thử bao gồm schema hợp lệ/sai, file không header giữ dòng đầu tiên,
header đảo thứ tự, UTF-8 BOM, MSE/risk khớp kết quả mô hình trước đó và luồng
Streamlit dữ liệu mẫu → phân tích → bảng/biểu đồ → lọc cảnh báo.

Code: [app.py](../app.py), [dashboard.py](../src/nsl_kdd/dashboard.py).
