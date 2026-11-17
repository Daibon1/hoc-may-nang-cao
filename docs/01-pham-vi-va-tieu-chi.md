# Giai đoạn 1: Phạm vi và tiêu chí đánh giá

## 1. Mục tiêu

Xây dựng và đánh giá một mô hình Autoencoder học biểu diễn lưu lượng bình thường trên NSL-KDD. Mô hình dùng sai số tái tạo để tính điểm bất thường, phát hiện lưu lượng đáng ngờ và xếp điểm thành ba mức rủi ro thấp, trung bình, cao.

Đề tài đồng thời so sánh Autoencoder với Isolation Forest, One-Class SVM và Random Forest. Kết quả cần thể hiện cả hiệu năng tổng thể lẫn khả năng phát hiện theo nhóm DoS, Probe, R2L và U2R.

## 2. Phạm vi

### Trong phạm vi

- Dùng các tập NSL-KDD `KDDTrain+` và `KDDTest+`.
- Xử lý 41 đặc trưng đầu vào; không dùng nhãn tấn công hoặc cột difficulty làm đặc trưng.
- Autoencoder được huấn luyện trên các mẫu bình thường của tập train.
- Chọn ngưỡng từ validation tách ra từ tập train; chỉ dùng tập test để đánh giá cuối cùng.
- Đánh giá phát hiện nhị phân bình thường/tấn công và phân tích riêng bốn nhóm DoS, Probe, R2L, U2R.
- So sánh với Isolation Forest, One-Class SVM và Random Forest.
- Báo cáo hiệu năng, tỷ lệ cảnh báo nhầm, cách chọn ngưỡng và giới hạn của bộ dữ liệu.

### Ngoài phạm vi

- Triển khai cảm biến thu thập lưu lượng mạng trực tiếp hoặc hệ thống cảnh báo production.
- Khẳng định điểm bất thường là xác suất tấn công hay mức thiệt hại thực tế.
- Tối ưu cho dữ liệu mạng hiện đại ngoài NSL-KDD trong giai đoạn thực nghiệm này.

## 3. Quy ước dữ liệu và chia tập

- Dùng `KDDTrain+` làm nguồn huấn luyện và validation; giữ nguyên `KDDTest+` làm tập đánh giá cuối.
- Tách validation từ `KDDTrain+` trước khi fit bộ tiền xử lý. Các thống kê chuẩn hóa và danh mục mã hóa chỉ được fit trên phần train.
- Autoencoder, Isolation Forest và One-Class SVM được huấn luyện bằng mẫu bình thường trong phần train.
- Random Forest là baseline có giám sát: được huấn luyện trên nhãn của phần train. Kết quả của nó sẽ được ghi nhận là so sánh có giám sát, không xem là cùng giả định học với các mô hình one-class.
- Mọi mô hình được đánh giá trên cùng tập test và cùng định nghĩa nhãn. Danh mục chưa thấy khi huấn luyện phải được xử lý rõ ràng, không làm rò rỉ thông tin từ test.
- Nhãn bình thường được giữ là `normal`; các nhãn tấn công được ánh xạ thành `DoS`, `Probe`, `R2L` hoặc `U2R` theo taxonomy NSL-KDD. Khi tính chỉ số nhị phân, bốn nhóm này gộp thành `attack`.

## 4. Định nghĩa điểm và mức rủi ro

### Điểm Autoencoder

Với mỗi mẫu, điểm bất thường ban đầu là mean squared error giữa vector đầu vào đã tiền xử lý và vector tái tạo. Sai số được tính trên cùng tập đặc trưng của mọi mẫu. Điểm cao hơn biểu thị mẫu khác với các mẫu bình thường mà mô hình đã học.

Để dễ diễn giải, có thể chuyển sai số thành **điểm phần trăm bất thường** từ 0 đến 100 bằng phân vị thực nghiệm so với sai số trên các mẫu bình thường của validation. Đây là thứ hạng tương đối trong phân phối tham chiếu, **không phải xác suất mẫu là tấn công**.

### Ngưỡng khởi đầu

Các phân vị dưới đây là quy ước khởi đầu cho thực nghiệm; sẽ được tính chỉ từ sai số của mẫu bình thường trong validation:

| Phân vị sai số bình thường | Mức hiển thị | Diễn giải |
|---|---|---|
| Dưới P95 | Thấp | Sai số chưa thuộc vùng bất thường cao theo tập tham chiếu |
| Từ P95 đến dưới P99 | Trung bình | Cần chú ý; vùng cảnh báo sớm |
| Từ P99 trở lên | Cao | Vùng cảnh báo chính; ngưỡng phát hiện nhị phân ban đầu |

Ngưỡng phát hiện ban đầu là P99, tương ứng mục tiêu tỷ lệ cảnh báo nhầm xấp xỉ 1% trên dữ liệu bình thường nếu phân phối validation đại diện. Ngưỡng và tỷ lệ cảnh báo nhầm thực tế phải được báo cáo; không giả định chúng giữ nguyên trên test hoặc ngoài thực tế.

Mức rủi ro mô tả độ bất thường theo Autoencoder, không khẳng định mức độ nguy hiểm hay tác động của cuộc tấn công. Bảng kết quả sẽ báo riêng nhãn thật để tránh nhầm điểm mô hình với mức độ nghiêm trọng thực tế.

## 5. Chỉ số đánh giá

### Chỉ số chính

- PR-AUC cho bài toán phát hiện tấn công, vì dữ liệu mất cân bằng lớp.
- Recall của tấn công tại ngưỡng phát hiện đã chọn trên validation.
- False Positive Rate (FPR) trên lưu lượng bình thường.
- Macro-F1 cho bài toán bình thường/tấn công; kèm Precision, Recall và F1 để có thể đọc rõ đánh đổi.

### Phân tích bổ sung

- ROC-AUC và confusion matrix trên test.
- Precision, Recall và F1 cho từng nhóm DoS, Probe, R2L và U2R.
- Phân bố số mẫu và tỷ lệ phát hiện ở từng mức thấp/trung bình/cao theo nhãn thật.
- Với mọi chỉ số theo nhóm, nêu rõ số mẫu hỗ trợ; không kết luận chắc chắn từ nhóm quá ít mẫu.

Ngưỡng vận hành được chọn từ validation và giữ cố định trên test. Đường cong AUC có thể dùng điểm liên tục, không cần ngưỡng, nhưng không thay thế kết quả tại ngưỡng vận hành.

## 6. Baseline và nguyên tắc so sánh

- **Isolation Forest, One-Class SVM:** so sánh theo hướng phát hiện bất thường; dùng mẫu bình thường trong train để fit và chọn ngưỡng theo validation.
- **Random Forest:** baseline phân loại có giám sát; báo riêng giả định được dùng nhãn train.
- Tất cả mô hình dùng cùng quy tắc chia train/validation/test và cùng tập đặc trưng đầu vào khi tương thích.
- Với mỗi mô hình, quy ước chiều điểm sao cho giá trị cao hơn luôn có nghĩa là đáng ngờ hơn; chọn ngưỡng từ validation, không điều chỉnh theo test.
- Báo cáo cấu hình, seed, thời gian huấn luyện/suy luận nếu đo được. Không tuyên bố Autoencoder tốt hơn trước khi có kết quả.

## 7. Tìm hiểu thuật toán và công nghệ

### Thuật toán được khảo sát

| Thuật toán | Cách hoạt động trong đề tài | Vai trò và điểm cần lưu ý |
|---|---|---|
| **Autoencoder** | Mạng encoder–decoder học tái tạo vector đầu vào. Huấn luyện bằng mẫu bình thường; sai số tái tạo lớn được xem là dấu hiệu khác thường. | Phương pháp chính để tạo anomaly score liên tục. Điểm số đo độ khác với dữ liệu học được, không phải xác suất tấn công; kiến trúc, regularization và ngưỡng cần kiểm chứng trên validation. |
| **Isolation Forest** | Tạo các cây phân hoạch ngẫu nhiên; điểm bất thường dựa trên việc một mẫu bị cô lập nhanh đến mức nào. | Baseline không giám sát/one-class, không cần nhãn tấn công để fit; chiều score phải quy đổi sao cho giá trị cao nghĩa là đáng ngờ. |
| **One-Class SVM** | Học một biên bao quanh phần lớn mẫu bình thường trong không gian đặc trưng; mẫu nằm ngoài biên bị xem là bất thường. | Baseline one-class. Nhạy với kernel, `nu` và scale đặc trưng; cần chuẩn hóa dữ liệu và chọn cấu hình mà không dùng test. |
| **Random Forest** | Tổ hợp nhiều cây quyết định được huấn luyện từ dữ liệu có nhãn. | Baseline phân loại có giám sát. Khác giả định với các mô hình học từ normal-only; dùng nhãn train và báo riêng kết quả để so sánh công bằng. |

### Công nghệ dự kiến

| Công nghệ | Công dụng dự kiến | Trạng thái |
|---|---|---|
| **Python** | Ngôn ngữ triển khai pipeline và thực nghiệm. Workspace hiện dùng Python 3.14.5. | Đã dùng cho script khảo sát dữ liệu; tương thích thư viện ML phải xác nhận trước khi cài. |
| **Pandas / NumPy** | Đọc bảng dữ liệu, biến đổi và thao tác vector/số liệu. | Đã khảo sát lựa chọn; chưa thêm vào môi trường dự án. |
| **scikit-learn** | Tiền xử lý, chia tập, Isolation Forest, One-Class SVM, Random Forest và metrics. | Công nghệ dự kiến; chưa cài đặt hoặc chạy mô hình. |
| **PyTorch** | Xây dựng và huấn luyện mạng Autoencoder. | Framework học sâu dự kiến; cần xác nhận phiên bản hỗ trợ interpreter/GPU trước khi cài. |
| **Mermaid** | Sơ đồ và biểu đồ khảo sát nhúng trực tiếp trong Markdown. | Dùng để trình bày kiểm tra dữ liệu; không cần xuất ảnh rời. Matplotlib/Seaborn có thể được cân nhắc cho biểu đồ kết quả mô hình sau này. |

Các công nghệ ở trạng thái “dự kiến” là lựa chọn thiết kế dựa trên vai trò cần
có, không phải thư viện đã cài hay hệ thống đã triển khai. Việc kiểm tra phiên
bản tương thích và khai báo dependencies sẽ thực hiện trước giai đoạn tiền xử
lý/huấn luyện.

## 8. Tiêu chí hoàn thành giai đoạn 1

Giai đoạn này được xem là hoàn thành khi nhóm thống nhất được:

1. Bài toán, phạm vi và giả định học của từng mô hình.
2. Cách chia dữ liệu, ngăn rò rỉ và ánh xạ nhãn.
3. Công thức điểm bất thường, cách đặt ngưỡng và ý nghĩa của ba mức rủi ro.
4. Bộ chỉ số chính, phân tích theo nhóm tấn công và nguyên tắc so sánh baseline.
5. Vai trò, giả định và giới hạn của Autoencoder cùng các thuật toán baseline.
6. Công nghệ dự kiến cho tiền xử lý, mô hình học sâu, baseline và trực quan hóa.

Đây là tiêu chí và quy ước thực nghiệm, chưa phải kết quả đạt được. Các ngưỡng P95/P99 cần được kiểm tra khi khảo sát dữ liệu và validation ở giai đoạn tiếp theo.

## 9. Giới hạn cần nêu trong báo cáo

NSL-KDD là bộ dữ liệu benchmark cũ, có phân phối và cách tạo dữ liệu riêng; kết quả tốt trên bộ này không tự chứng minh hiệu quả trên lưu lượng mạng hiện đại. Điểm reconstruction error đo mức khác biệt so với dữ liệu bình thường đã học, không xác định chắc chắn nguyên nhân hay mức độ tác hại. Kết quả theo R2L/U2R cần được diễn giải thận trọng do số mẫu có thể ít và mất cân bằng.
