"""Generate the four-model comparison report from measured results."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_report(root=ROOT):
    directory = root / 'reports/model_comparison'
    report = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
    lines = ['# So sánh bốn mô hình trên KDDTest+', '',
             f"Đánh giá trên cùng **{report['test_rows']:,} mẫu**, gồm {report['test_attack_rows']:,} Attack. "
             'Attack=1; score càng cao càng bất thường. Mọi ngưỡng được chốt bằng validation, '
             'không chọn hoặc điều chỉnh bằng test.', '',
             '| Mô hình | Giả định học | Accuracy | Precision Attack | Recall Attack | F1 Attack | Macro-F1 | FPR | PR-AUC (AP) | ROC-AUC |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, result in report['test_metrics'].items():
        policy = 'Có giám sát' if name == 'Random Forest' else 'Normal-only'
        values = [f"{result[key]:.2%}" for key in ['accuracy', 'precision_attack', 'recall_attack',
                                                  'f1_attack', 'macro_f1', 'false_positive_rate']]
        lines.append(f"| {name} | {policy} | " + ' | '.join(values) +
                     f" | {result['pr_auc_average_precision']:.4f} | {result['roc_auc']:.4f} |")
    lines += ['', '## Cách đọc và giới hạn so sánh', '',
              '- Random Forest được dùng nhãn Normal/Attack của toàn bộ phần train. Ba mô hình còn lại '
              'chỉ dùng Normal để học trọng số/cấu trúc. Không xem đây là so sánh cùng giả định học.',
              '- Cùng 77 đặc trưng từ pipeline fit trên Normal train. Không fit lại preprocessing trên test.',
              '- Autoencoder giữ nguyên latent dimension và threshold đã tối ưu. Không huấn luyện lại '
              'hoặc chọn lại bằng kết quả của ba baseline.',
              '- Cùng Normal calibration và validation selection với Autoencoder: '
              f"{report['validation_calibration_normal_rows']} và {report['validation_selection_rows']} mẫu tương ứng. "
              'Các ngưỡng P97/P98/P99/P99.5/P99.9 tính từ Normal calibration; chọn macro-F1 selection '
              'cao nhất với FPR<=1%. Nếu không có ứng viên đạt, chọn FPR thấp nhất và báo constraint không đạt.',
              '- Nhãn Attack validation có dùng để chọn threshold; huấn luyện IF/OCSVM/AE vẫn là Normal-only. '
              'Ngân sách FPR validation không bảo đảm FPR test <=1%.',
              '- PR-AUC báo theo average precision từ score liên tục gốc. F1, recall, FPR phụ thuộc '
              'threshold. Không so giá trị score hoặc threshold tuyệt đối giữa các mô hình.',
              '- Baseline dùng cấu hình định trước, chưa có tìm kiếm siêu tham số đa cấu hình. '
              'AE đã được thử latent 8/16/32, nên ngân sách tối ưu chưa bằng nhau. Một seed, một benchmark '
              'không chứng minh ưu thế tổng quát.',
              '- Test đã được xem ở các giai đoạn trước. Lần so sánh này cố định tất cả cấu hình/ngưỡng '
              'trước khi tính test; chưa có tập đánh giá ngoài NSL-KDD.', '',
              '## Ngưỡng và cấu hình', '',
              '| Mô hình | Mẫu huấn luyện | Threshold | FPR selection <=1% đạt? |',
              '|---|---:|---:|---|']
    for name, metadata in report['model_metadata'].items():
        lines.append(f"| {name} | {metadata['training_rows']} | {metadata['threshold']:.8g} | "
                     f"{'Có' if metadata['selection_fpr_budget_met'] else 'Không'} |")
    lines += ['', '- RF: 300 cây, class_weight=balanced, seed 42, n_jobs=4.',
              '- IF: 300 cây, max_samples=256/cây, fit trên toàn bộ Normal train, seed 42. '
              'Score = -score_samples; không dùng offset contamination để chọn ngưỡng.',
              '- OCSVM: RBF, gamma=scale, nu=0.05, cache_size=1024 MB, không giới hạn số iteration. '
              'Fit toàn bộ Normal train, không lấy mẫu con. Score = -decision_function; có thể âm.',
              '- AE: checkpoint tối ưu latent 16 hiện có; score = reconstruction MSE.', '',
              '## Recall theo nhóm tấn công và cảnh báo nhầm', '',
              '| Mô hình | DoS | Probe | R2L | U2R | FPR Normal |', '|---|---:|---:|---:|---:|---:|']
    for name, result in report['test_metrics'].items():
        rates = [f"{result['groups'][group]['flagged_attack_rate']:.2%}" for group in ['DoS', 'Probe', 'R2L', 'U2R', 'normal']]
        lines.append(f"| {name} | " + ' | '.join(rates) + ' |')
    lines += ['', 'Các tỷ lệ trên dùng cùng số mẫu hỗ trợ theo nhóm. Nhóm U2R nhỏ; '
              'không nên kết luận chắc chắn từ tỷ lệ của một nhóm ít mẫu.', '',
              '## Confusion matrix', '',
              'Hàng là nhãn thật, cột là dự đoán, thứ tự `[Normal, Attack]`.', '']
    for name, result in report['test_metrics'].items():
        lines.append(f"- {name}: `{result['confusion_matrix']}`.")
    lines += ['', '## Nhận xét từ kết quả', '']
    for key, label in [('pr_auc_average_precision', 'PR-AUC'), ('macro_f1', 'macro-F1'), ('recall_attack', 'recall Attack')]:
        name = max(report['test_metrics'], key=lambda model: report['test_metrics'][model][key])
        lines.append(f"- {label} cao nhất trong lần chạy: **{name}**, {report['test_metrics'][name][key]:.4f}.")
    name = min(report['test_metrics'], key=lambda model: report['test_metrics'][model]['false_positive_rate'])
    lines += [f"- FPR thấp nhất: **{name}**, {report['test_metrics'][name]['false_positive_rate']:.2%}.", '',
              'Các nhận xét là mô tả test, không dùng để chọn lại cấu hình hoặc ngưỡng. '
              'Cần đọc đồng thời recall và FPR, đặc biệt khi so sánh RF có giám sát với mô hình Normal-only.', '',
              '## Biểu đồ và file', '', '![PR và ROC trên cùng KDDTest+](comparison_curves.png)', '',
              '- [Bảng CSV bốn mô hình](comparison.csv)',
              '- [Toàn bộ metrics, cấu hình và hash dữ liệu](metrics.json)',
              '- [Phân tích threshold validation](validation_thresholds.csv)',
              '- [Tỷ lệ phát hiện theo nhóm và số mẫu hỗ trợ](group_detection.csv)',
              '- [Hướng dẫn chạy lại](../../docs/06-model-comparison.md)', '']
    path = directory / 'comparison.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


if __name__ == '__main__':
    print(build_report())
