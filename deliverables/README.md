# Bộ nộp và thuyết trình

- [Báo cáo Word](final_report.docx) · [Bản Markdown](final_report.md)
- [Slide PowerPoint, có speaker notes](presentation.pptx)
- [Kịch bản demo và hỏi đáp](demo_script.md)
- [Kiểm thử tích hợp pipeline](pipeline_audit.json)
- [Bảng so sánh bốn mô hình](../reports/model_comparison/comparison.md)
- [Hướng dẫn chạy dashboard](../docs/07-streamlit-demo.md)
- [ZIP toàn dự án](nsl_kdd_submission.zip) · [Manifest SHA-256](manifest.json)

ZIP chứa code, notebook có output, raw/processed data, mô hình đã huấn luyện,
pipeline/reference/threshold, báo cáo và slide. Không chứa `.venv`, cache hoặc
cấu hình IDE cá nhân. Cài dependency từ `requirements-lock.txt` trên máy mới.

Đã kiểm tra 21 unit/UI tests đạt; kiểm thử tích hợp dựng lại preprocessing và
tái lập score của bốn estimator đã lưu trên đủ 22.544 mẫu test. Số liệu giữ nguyên,
không chọn lại cấu hình theo test khi chốt bộ nộp.

Lệnh dựng lại tài liệu sau khi có model/kết quả:

```powershell
./.venv/Scripts/python.exe scripts/audit_submission.py
./.venv/Scripts/python.exe scripts/build_submission.py
```

Muốn tái lập huấn luyện từ đầu, theo thứ tự preprocessing → train/optimize →
compare_models → audit_submission → build_submission như các tài liệu trong docs.
Huấn luyện lại có thể tạo file nhị phân/hash mới dù số liệu giống; cần chạy lại
các bước phụ thuộc để provenance khớp nhau.
