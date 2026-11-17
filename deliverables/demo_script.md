# Kịch bản thuyết trình và demo

## Thời lượng dự kiến

- Slide 1–6: 4 phút, bài toán, dữ liệu, validation, kiến trúc và scoring.
- Slide 7–11: 4 phút, latent/threshold, so sánh và phân tích nhóm.
- Slide 12: demo trực tiếp khoảng 3 phút.
- Slide 13 là phương án dự phòng; slide 14–15 kết luận và giới hạn khoảng 1 phút.
- Chừa 3–5 phút hỏi đáp. Slide PowerPoint có speaker notes cho từng trang.

## Chuẩn bị trước buổi trình bày

1. Giải nén bộ nộp vào một thư mục, mở project trong VS Code.
2. Máy đã có môi trường: nhấp đúp `run_demo.cmd` hoặc chạy:

```powershell
./.venv/Scripts/python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

3. Máy mới: cài Python 3.14 tương thích, tạo `.venv`, cài `requirements-lock.txt`
   trước buổi demo khi có Internet. Không mang/copy `.venv` giữa các máy.
4. Mở http://127.0.0.1:8501, để terminal còn chạy. Không huấn luyện lại trong buổi demo.
5. Mở sẵn [slide](presentation.pptx), [báo cáo](final_report.docx),
   [bảng so sánh](../reports/model_comparison/comparison.md) và file raw test.
6. Chạy kiểm tra nhanh từ thư mục project:

```powershell
./.venv/Scripts/python.exe scripts/verify_model_comparison.py
```

## Demo trực tiếp: từng thao tác và lời dẫn

| Thời gian | Thao tác | Lời dẫn / kết quả mong đợi |
|---|---|---|
| 0:00–0:30 | Chọn “Dùng 100 dòng mẫu”, nhấn “Phân tích rủi ro” | “Ứng dụng nạp model đã huấn luyện, không học lại từ dữ liệu upload.” Schema hợp lệ, bảng/biểu đồ xuất hiện. |
| 0:30–1:10 | Bỏ chọn mẫu, upload `data/raw/KDDTest+.txt` ở chế độ Tự nhận diện | “File không header vẫn giữ đủ dòng đầu tiên; label/difficulty không được đưa vào mô hình.” Có 22.544 dòng hợp lệ. |
| 1:10–1:40 | Nhấn phân tích, chỉ biểu đồ và các thẻ thống kê | High 7.453, Medium 3.958, Low 11.133. “Tỷ lệ High là tỷ lệ dự đoán, không phải độ chính xác.” |
| 1:40–2:10 | Lọc chỉ High, xem Risk Score và MSE | “Điểm là phân vị Normal tham chiếu. Hai mẫu có thể cùng điểm 100 nhưng MSE khác nhau.” |
| 2:10–2:30 | Nhấn “Tải toàn bộ kết quả CSV”, mở CSV | 22.544 dòng và header, có row_id, error, risk, alert, prediction, original_label. Bộ lọc không cắt file tải. |
| 2:30–3:00 | Upload CSV lỗi [invalid_missing_service.csv](demo_inputs/invalid_missing_service.csv) | Hiển thị thiếu cột service, không dự đoán. Chuyển lại file hợp lệ để kết thúc. |

Nếu dữ liệu upload có category mới, chỉ rõ cảnh báo mã hóa block 0; không khẳng định
category lạ tự động là tấn công. Không chỉnh threshold để làm demo “đẹp hơn”.

## Phương án dự phòng

- Server không mở: kiểm tra terminal; dùng đúng `.venv/Scripts/python.exe -m streamlit`,
  không dùng lệnh `streamlit` trần. Port bận: đổi `--server.port 8502` và mở URL tương ứng.
- Thiếu thư viện trên máy mới: dùng máy đã chuẩn bị hoặc trình bày slide 13 cùng
  [kết quả dự phòng](figures/demo_counts.png) và CSV đã xuất. Không cài/thử huấn luyện
  kéo dài trước hội đồng.
- Thiếu model: cần nguyên thư mục `artifacts/models/autoencoder_optimized/`; không
  chỉ copy một mình model.pt. Bản ZIP đã kèm các artifact, chưa kèm môi trường Python.
- Browser chặn tải file: dùng các file CSV kết quả đã có trong reports để giải thích
  định dạng; phân biệt dữ liệu dự phòng đã đo với thao tác live chưa hoàn tất.

## Câu hỏi thường gặp và trả lời ngắn

**Vì sao Autoencoder chỉ học Normal?**
Để học cấu trúc bình thường, sau đó dùng sai số tái tạo làm mức bất thường.
Mô hình vẫn có thể tái tạo tốt một số Attack nên không bảo đảm phát hiện hết.

**Score 99 có nghĩa là 99% khả năng bị tấn công?**
Không. Nó là phân vị tương đối với sai số Normal calibration. Điểm không phải xác suất.

**Vì sao chọn latent 16 thay vì 32 có MSE thấp hơn?**
Tiêu chí chính đã định là AP validation selection. 16 đạt 0,978826 so với 32 là
0,978095. Chênh lệch nhỏ và chỉ có một seed nên chưa kết luận ý nghĩa thống kê.

**Ngưỡng P99 có bảo đảm FPR 1% ngoài thực tế?**
Không. FPR selection khoảng 0,99%, test 1,51%; thay đổi phân phối có thể làm khác nữa.

**Random Forest tốt hơn Autoencoder?**
RF có recall/macro-F1 tốt hơn trong lần chạy này nhưng được dùng nhãn train.
AE có AP cao hơn; không tuyên bố một mô hình tốt hơn trên mọi tiêu chí hoặc cùng giả định học.

**Nhóm nào yếu nhất? Vì sao?**
AE yếu ở R2L: 495/2.885, recall 17,16%. Bảng này chứng minh hiện tượng, chưa chứng minh
nguyên nhân; cần phân tích false negatives và đặc trưng. U2R chỉ 67 mẫu nên biến động lớn.

**Có rò rỉ nhãn không?**
Nhãn/difficulty không vào features. Encoder/scaler chỉ fit trên Normal train.
Nhưng có dùng nhãn validation để chọn latent và threshold, nên không gọi toàn bộ
quy trình chọn cấu hình là hoàn toàn không giám sát. Test từng được xem trong dự án.

**Đã kiểm thử toàn pipeline như thế nào?**
21 unit/UI tests; dựng lại preprocessing từ raw và đối chiếu toàn bộ ma trận; nạp
bốn model và tái lập tất cả score/nhãn test; kiểm tra risk, alert, schema, upload
và endpoint download. Audit cuối không huấn luyện lại mô hình.

**Có thể dùng production chưa?**
Chưa. Đây là benchmark cũ và demo bản ghi kết nối. Cần dữ liệu hiện đại, đánh giá
drift, chi phí false alarms, giám sát vận hành và kiểm chứng ngoài benchmark.
