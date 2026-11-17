"""Render experiment tables and overfitting observations from saved measurements."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def build_report(root=ROOT):
    directory = root / 'reports/risk_optimization'
    report = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    lines = ['# Kết quả risk scoring và tối ưu', '',
             '## So sánh latent dimension trên validation selection', '',
             '| Latent | Epoch tốt nhất / thực chạy | MSE train | MSE early stopping | AP selection |',
             '|---|---:|---:|---:|---:|']
    for trial in report['trials']:
        lines.append(f"| {trial['latent_dim']} | {trial['best_epoch']} / {trial['epochs_run']} | "
                     f"{trial['train_mse']:.6f} | {trial['early_stopping_mse']:.6f} | "
                     f"{trial['validation_selection_ap']:.6f} |")
    lines += ['', f"Chọn latent **{report['selected_latent_dim']}** theo AP selection cao nhất. "
              'Không chọn bằng MSE train hoặc kết quả test. Chênh lệch AP giữa 16 và 32 nhỏ; '
              'chưa có nhiều seed để xác nhận khác biệt có ý nghĩa thống kê.', '',
              '## Threshold và cảnh báo', '',
              '| Phân vị calibration | MSE threshold | Precision selection | Recall selection | FPR selection | Macro-F1 |',
              '|---|---:|---:|---:|---:|---:|']
    thresholds = pd.read_csv(directory / 'threshold_analysis.csv')
    for row in thresholds.to_dict('records'):
        lines.append(f"| P{100 * row['quantile']:g} | {row['threshold']:.6f} | "
                     f"{row['precision_attack']:.2%} | {row['recall_attack']:.2%} | "
                     f"{row['false_positive_rate']:.2%} | {row['macro_f1']:.4f} |")
    risk = report['risk']
    lines += ['', f"Ngưỡng High đã chọn: P{100 * risk['high_quantile']:g} = **{risk['high_threshold']:.8f}**. "
              f"Ngân sách FPR selection <=1% đạt: **{report['selection_fpr_budget_met']}**.", '',
              f"- Low: MSE < {risk['medium_threshold']:.8f}.",
              f"- Medium: {risk['medium_threshold']:.8f} <= MSE < {risk['high_threshold']:.8f}.",
              f"- High: MSE >= {risk['high_threshold']:.8f}; tương ứng dự đoán Attack.", '',
              'Điểm 0–100 là empirical CDF của Normal calibration, không phải xác suất tấn công. '
              'Mức cảnh báo dùng MSE chưa làm tròn để tránh sai ranh giới của điểm hiển thị.', '',
              'P95 có recall cao hơn nhưng FPR selection vượt ngân sách. Nâng ngưỡng giảm cảnh báo '
              'nhầm và đồng thời bỏ sót thêm tấn công. Không dùng test để đổi ngưỡng.', '',
              '## Overfitting', '']
    for trial in report['trials']:
        dim = trial['latent_dim']
        history = pd.read_csv(directory / f'history_latent_{dim}.csv')
        last = history.iloc[-1]
        deterioration = last.early_stopping_mse / trial['early_stopping_mse'] - 1
        lines.append(f"- Latent {dim}: MSE stop/train của checkpoint tốt nhất = "
                     f"{trial['stop_train_ratio']:.3f}; MSE stop epoch cuối cao hơn tốt nhất "
                     f"{deterioration:.1%}. Đã khôi phục epoch {trial['best_epoch']}.")
    lines += ['', 'Loss dao động và có giai đoạn validation xấu đi, nên checkpoint cuối không '
              'được chọn tự động. Không có khoảng chênh train/validation lớn ở checkpoint tốt nhất '
              'trong ba thử nghiệm; điều này chưa đủ chứng minh mô hình không overfit. '
              'Latent 32 giảm MSE nhiều hơn nhưng AP không cao nhất, cho thấy khả năng tái tạo '
              'Normal tốt hơn không tự bảo đảm phát hiện Attack tốt hơn. Cấu hình 32 chạm giới hạn '
              '100 epoch, chưa kết luận hội tụ hoàn toàn. Chỉ chạy một seed mỗi cấu hình.', '',
              '## Đánh giá test sau khi chốt cấu hình', '',
              '| Chỉ số | Giá trị |', '|---|---:|']
    test = report['test']
    for name in ['precision_attack', 'recall_attack', 'false_positive_rate', 'f1_attack',
                 'macro_f1', 'pr_auc_average_precision', 'roc_auc']:
        lines.append(f"| {name} | {test[name]:.6f} |")
    lines += ['', f"Confusion matrix (hàng thật, cột dự đoán; Normal/Attack): `{test['confusion_matrix']}`.", '',
              '| Nhóm | Số mẫu | Tỷ lệ bị gán Attack |', '|---|---:|---:|']
    for group, values in test['groups'].items():
        lines.append(f"| {group} | {values['support']} | {values['flagged_attack_rate']:.2%} |")
    lines += ['', 'Tỷ lệ gán Attack là recall đối với nhóm tấn công và FPR đối với Normal. '
              'R2L vẫn bị bỏ sót nhiều; U2R có ít mẫu. Mục tiêu FPR <=1% trên validation '
              f"selection không bảo đảm trên test: FPR test thực tế = {test['false_positive_rate']:.2%}.", '']
    baseline_path = root / 'reports/autoencoder/metrics.json'
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding='utf-8'))['evaluation']['test']
        lines += ['## So với giai đoạn trước (chỉ mô tả)', '',
                  '| Chỉ số test | Trước | Tối ưu |', '|---|---:|---:|']
        for metric in ['precision_attack', 'recall_attack', 'false_positive_rate', 'macro_f1']:
            lines.append(f"| {metric} | {baseline[metric]:.4f} | {test[metric]:.4f} |")
        lines += ['', 'Hai giai đoạn có cách chia vai trò validation khác nhau; không quy toàn bộ '
                  'chênh lệch cho latent dimension. Đây là đánh đổi recall/FPR, không phải cải thiện '
                  'đồng thời mọi chỉ số. Test đã được xem ở giai đoạn trước; lần tìm kiếm này không '
                  'dùng test để chọn model hay threshold.', '']
    lines += ['## Biểu đồ và đầu ra', '', '![So sánh loss, threshold và risk score](optimization.png)', '',
              '- [Dự đoán test kèm risk score](test_risk_predictions.csv)',
              '- [Phân bố mức cảnh báo theo nhóm](risk_by_group.csv)',
              '- [Số liệu đầy đủ và giao thức](summary.json)',
              '- [Hướng dẫn chạy](../../docs/05-risk-scoring.md)', '']
    path = directory / 'analysis.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


if __name__ == '__main__':
    print(build_report())
