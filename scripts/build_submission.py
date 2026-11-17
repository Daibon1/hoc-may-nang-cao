"""Build measured final report, PowerPoint, demo script, figures and submission archive."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Inches as DocInches, Pt as DocPt
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'deliverables'
FIG = OUT / 'figures'
MODELS = ['Autoencoder', 'Random Forest', 'Isolation Forest', 'One-Class SVM']
GROUPS = ['DoS', 'Probe', 'R2L', 'U2R']


def md_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def build():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    comparison = json.loads((ROOT / 'reports/model_comparison/metrics.json').read_text(encoding='utf-8'))
    optimization = json.loads((ROOT / 'reports/risk_optimization/summary.json').read_text(encoding='utf-8'))
    audit = json.loads((OUT / 'pipeline_audit.json').read_text(encoding='utf-8'))
    assert audit['status'] == 'PASS'
    results = comparison['test_metrics']
    ae = results['Autoencoder']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white', 'font.size': 11})
    heat = np.array([[results[name]['groups'][group]['flagged_attack_rate'] for group in GROUPS] for name in MODELS])
    fig, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(heat, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(range(4), [f'{group}\nn={ae["groups"][group]["support"]:,}' for group in GROUPS])
    ax.set_yticks(range(4), MODELS)
    for row in range(4):
        for col in range(4):
            ax.text(col, row, f'{heat[row, col]:.1%}', ha='center', va='center',
                    color='white' if heat[row, col] > .6 else '#102c41', weight='bold')
    ax.set_title('Recall theo nhóm tấn công · cùng KDDTest+')
    fig.colorbar(image, ax=ax, label='Recall'); fig.tight_layout()
    fig.savefig(FIG / 'group_recall.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    matrix = np.array(ae['confusion_matrix'])
    axes[0].imshow(matrix, cmap='Blues')
    axes[0].set_xticks([0, 1], ['Normal', 'Attack']); axes[0].set_yticks([0, 1], ['Normal', 'Attack'])
    axes[0].set(xlabel='Dự đoán', ylabel='Nhãn thật', title='Autoencoder · confusion matrix')
    for r in range(2):
        for c in range(2):
            axes[0].text(c, r, f'{matrix[r,c]:,}', ha='center', va='center',
                         color='white' if matrix[r,c] > matrix.max()*.6 else '#102c41', fontsize=18)
    supports = [ae['groups'][g]['support'] for g in GROUPS]
    detected = [round(ae['groups'][g]['flagged_attack_rate'] * ae['groups'][g]['support']) for g in GROUPS]
    missed = np.array(supports) - detected
    axes[1].bar(GROUPS, detected, label='Phát hiện', color='#14967c')
    axes[1].bar(GROUPS, missed, bottom=detected, label='Bỏ sót', color='#e05260')
    axes[1].set(title='Autoencoder · số phát hiện / bỏ sót', ylabel='Số mẫu'); axes[1].legend()
    fig.tight_layout(); fig.savefig(FIG / 'ae_errors.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    risk = pd.read_csv(ROOT / 'reports/risk_optimization/test_risk_predictions.csv')
    counts = risk.alert_level.value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0)
    bars = ax.bar(counts.index, counts.values, color=['#14967c', '#e6a23c', '#e05260'])
    ax.bar_label(bars, labels=[f'{n:,}' for n in counts.values], padding=4)
    ax.set(title='Demo dự phòng · kết quả đủ 22.544 mẫu test', ylabel='Số mẫu'); ax.set_ylim(0, counts.max()*1.18)
    fig.tight_layout(); fig.savefig(FIG / 'demo_counts.png', dpi=180); plt.close(fig)
    shutil.copy2(ROOT / 'reports/risk_optimization/optimization.png', FIG / 'optimization.png')
    shutil.copy2(ROOT / 'reports/model_comparison/comparison_curves.png', FIG / 'comparison_curves.png')

    table_rows = [[name, f"{results[name]['accuracy']:.2%}", f"{results[name]['precision_attack']:.2%}",
                   f"{results[name]['recall_attack']:.2%}", f"{results[name]['macro_f1']:.4f}",
                   f"{results[name]['false_positive_rate']:.2%}", f"{results[name]['pr_auc_average_precision']:.4f}"] for name in MODELS]
    group_rows = []
    for group, support, tp, fn in zip(GROUPS, supports, detected, missed):
        group_rows.append([group, support, tp, int(fn), f"{ae['groups'][group]['flagged_attack_rate']:.2%}"])
    sections = [
        ('Tóm tắt', 'Đề tài xây dựng Autoencoder phát hiện bất thường và chấm điểm lưu lượng NSL-KDD. '
         'Mô hình chỉ học từ Normal train; MSE tái tạo được hiệu chuẩn thành điểm phân vị 0–100 và ba mức cảnh báo. '
         'Latent 16 được chọn trong ba thử nghiệm bằng AP validation. Trên KDDTest+, AE đạt AP 0,9662, '
         'recall Attack 56,93%, FPR 1,51%; chưa đáp ứng đồng thời recall cao và cảnh báo nhầm thấp. '
         'Web demo Streamlit hỗ trợ upload, kiểm tra schema, biểu đồ và tải CSV.'),
        ('1. Bài toán và phạm vi', 'Mục tiêu gồm phân loại Normal/Attack, tạo điểm bất thường dễ đọc, '
         'phân tích DoS/Probe/R2L/U2R và so sánh ba baseline. Đầu vào là bản ghi kết nối NSL-KDD, '
         'không phải packet capture hoặc luồng mạng trực tiếp. Điểm rủi ro không đại diện xác suất '
         'tấn công, mức thiệt hại hoặc nhãn mức độ nguy hiểm đã được kiểm chứng.'),
        ('2. Dữ liệu và preprocessing', 'KDDTrain+ có 125.973 mẫu, KDDTest+ có 22.544 mẫu. '
         '41 đặc trưng gồm 38 numeric và ba categorical: protocol_type, service, flag. Loại nhãn và '
         'difficulty khỏi đầu vào. Split train/validation 80/20, stratify nhị phân, seed 42: 100.778/25.195 mẫu. '
         'Chỉ 53.874 Normal train được dùng fit OneHotEncoder/StandardScaler và huấn luyện AE, IF, OCSVM. '
         'Đầu ra có 77 chiều; category chưa thấy mã hóa block 0. Test có 1.343 dòng service lạ và hai dòng '
         'flag lạ; giới hạn này được cảnh báo trên dashboard. Dữ liệu gốc được kiểm tra hash theo mirror đã pin.'),
        ('3. Thiết kế thực nghiệm và tránh rò rỉ', 'Validation chia 40/30/30, stratify nhị phân: '
         '5.388 Normal early stopping, 4.040 Normal calibration và 7.559 mẫu selection có cả hai lớp. '
         'Chọn epoch bằng early-stopping loss; chọn latent bằng AP selection; chọn threshold từ phân vị '
         'Normal calibration, đánh giá macro-F1/FPR trên selection. Mục tiêu FPR selection <=1%; '
         'không bảo đảm FPR test <=1%. Có sử dụng nhãn Attack validation để chọn cấu hình/ngưỡng, '
         'nhưng không dùng Attack train để fit AE. Test đã được xem qua các giai đoạn trước; không gọi '
         'đây là đánh giá mù trên tập chưa từng quan sát. Không dùng test để chọn lại cấu hình trong lần tìm kiếm này.'),
        ('4. Autoencoder và risk scoring', 'Encoder 77→64→32→16; decoder 16→32→64→77. ReLU ở hidden, '
         'bottleneck/output linear phù hợp StandardScaler không bị chặn [0,1]. Adam lr=0,001, batch 256, '
         'weight decay=1e-5, clip norm=5, tối đa 100 epoch, patience=12. Score gốc là trung bình bình phương '
         'sai số trên 77 đặc trưng. Risk Score = 100 × số Normal calibration có MSE <= MSE mẫu / 4.040. '
         'Điểm bão hòa ngoài miền tham chiếu, nên giữ MSE gốc để so sánh các mẫu cùng điểm 100. '
         'Low: MSE<0,02376216; Medium: từ ngưỡng đó đến dưới 0,10899014; High: từ 0,10899014 trở lên. '
         'High là Attack; Low/Medium là Normal. Dùng giá trị chưa làm tròn để quyết định.'),
        ('5. Tối ưu latent, threshold và overfitting', 'AP selection cho latent 8/16/32 lần lượt '
         '0,972914 / 0,978826 / 0,978095. Chọn 16, checkpoint epoch 83/95 epoch thực chạy. '
         'Latent 32 có MSE Normal thấp hơn nhưng AP không cao nhất; tái tạo tốt hơn không tự bảo đảm '
         'phát hiện tốt hơn. Chênh AP 16–32 nhỏ, chưa có nhiều seed để kết luận khác biệt có ý nghĩa. '
         'P95 có recall selection 95,59% nhưng FPR 4,45%; P99 đạt recall 80,96%, FPR 0,99%, được chọn '
         'theo ngân sách. Ở checkpoint 16 tốt nhất, train MSE=0,01338, stop MSE=0,01088; loss stop cuối '
         'cao hơn tốt nhất 15,4%, nên đã khôi phục checkpoint tốt nhất. Dao động không đủ chứng minh '
         'overfitting đã bị loại bỏ. Cấu hình 32 chạm 100 epoch, chưa kết luận hội tụ hoàn toàn.'),
        ('6. So sánh bốn mô hình', 'RF dùng toàn bộ 100.778 mẫu train có nhãn, 300 cây, class_weight=balanced. '
         'IF dùng Normal train, 300 cây, max_samples=256 mỗi cây. OCSVM dùng toàn bộ Normal train, '
         'RBF gamma=scale, nu=0,05; kiểm tra hội tụ đạt. Cùng preprocessing, calibration/selection và test. '
         'Ngưỡng baseline chọn trên P97/P98/P99/P99.5/P99.9 với cùng quy tắc macro-F1/FPR. '
         'RF có giám sát nên không cùng giả định học với ba mô hình Normal-only. Baseline chưa được '
         'tối ưu nhiều cấu hình như AE; ngân sách thực nghiệm không ngang nhau.'),
        ('7. Phân tích từng nhóm tấn công', 'DoS: AE phát hiện 5.201/7.460 (69,72%), bỏ sót 2.259. '
         'Probe: 1.569/2.421 (64,81%), bỏ sót 852. R2L: 495/2.885 (17,16%), bỏ sót 2.390; đây là '
         'điểm yếu rõ nhất của AE tại ngưỡng hiện tại. U2R: 41/67 (61,19%), bỏ sót 26; một mẫu đã '
         'làm thay đổi recall khoảng 1,49 điểm phần trăm, nên không diễn giải quá mức. '
         'RF đạt recall DoS/Probe cao hơn (85,47%/87,28%) nhưng R2L/U2R thấp (8,94%/20,90%). '
         'IF rất yếu ở R2L (1,39%). OCSVM có R2L 18,65% và FPR thấp nhất, nhưng recall tổng chỉ '
         '48,45%. Không có mô hình tốt nhất đồng thời trên mọi nhóm và mọi tiêu chí. '
         'Sự giống nhau giữa mẫu tấn công và Normal là một giả thuyết giải thích cần kiểm chứng thêm, '
         'không phải nguyên nhân đã được chứng minh từ các bảng này.'),
        ('8. Kiểm thử và dashboard', '21 kiểm thử unit/UI đạt. Kiểm thử tích hợp đã dựng lại preprocessing '
         'từ raw, đối chiếu mọi ma trận và nhãn, kiểm tra validation rời nhau, nạp cả bốn estimator và '
         'tái lập toàn bộ dự đoán test. Đường raw CSV→schema→AE→MSE→Risk Score→mức cảnh báo khớp '
         '22.544 mẫu. Notebook có đủ output đã thực thi, không có lỗi. Kiểm thử cuối không huấn luyện '
         'lại các mô hình hoặc tìm kiếm siêu tham số. Browser demo đã xử lý full test và endpoint tải '
         'CSV trả HTTP 200 đủ 22.544 dòng. Dashboard giới hạn 20 MB/100.000 dòng, kiểm tra tên/thứ tự '
         'cột, dữ liệu thiếu, giá trị hữu hạn và miền cơ bản; hỗ trợ có/không header, lọc cảnh báo, '
         'biểu đồ và download toàn bộ dữ liệu theo thứ tự gốc.'),
        ('9. Kết luận và hướng phát triển', 'Đã hoàn thành pipeline tái lập, mô hình lưu/nạp, chấm điểm '
         'ba mức, so sánh và demo. AE có AP/ROC-AUC cao nhất trong lần chạy; RF có recall/macro-F1 '
         'cao nhất nhưng được dùng nhãn train; OCSVM có FPR thấp nhất. AE vẫn bỏ sót 5.527 Attack, '
         'cảnh báo nhầm 147 Normal; FPR test 1,51% vượt ngân sách validation 1%. Không tuyên bố '
         'mô hình đủ cho production. Hướng tiếp theo: nhiều seed, phân tích false negatives theo '
         'đặc trưng, log-transform/robust scaling có validation riêng, cân bằng trọng số MSE numeric '
         'và one-hot, baseline tuning với cùng ngân sách, đánh giá trên dữ liệu mạng mới. Mọi thay '
         'đổi cần tập đánh giá mới hoặc protocol đánh giá phù hợp, tránh tiếp tục tối ưu vào test hiện tại.'),
        ('Tài liệu tham khảo và nguồn tái lập', 'M. Tavallaee, E. Bagheri, W. Lu, A. Ghorbani, “A Detailed '
         'Analysis of the KDD CUP 99 Data Set”, CISDA 2009. Mirror: https://github.com/Jehuty4949/NSL_KDD, '
         'commit 27bbbdf5c5cccf9970b8b6037a0f19dc09c5a0a9. Số liệu trong báo cáo lấy từ '
         'reports/model_comparison/metrics.json và reports/risk_optimization/summary.json; '
         'audit nằm trong deliverables/pipeline_audit.json. Phiên bản chính xác trong requirements-lock.txt.')]

    markdown = ['# Xây dựng mô hình Autoencoder phát hiện và chấm điểm rủi ro bất thường trên lưu lượng mạng',
                '', '**Đánh giá trên NSL-KDD · Báo cáo thực nghiệm cuối kỳ**', '']
    doc = Document()
    doc.styles['Normal'].font.name = 'Calibri'
    doc.styles['Normal'].font.size = DocPt(11)
    doc.add_heading('Autoencoder phát hiện và chấm điểm rủi ro bất thường', 0)
    doc.add_paragraph('Đánh giá trên NSL-KDD · Báo cáo thực nghiệm cuối kỳ')
    for title, body in sections:
        markdown += ['## ' + title, '', body, '']
        doc.add_heading(title, 1); doc.add_paragraph(body)
        if title.startswith('5.'):
            picture = 'optimization.png'
        elif title.startswith('6.'):
            headers = ['Mô hình', 'Accuracy', 'Precision', 'Recall', 'Macro-F1', 'FPR', 'AP']
            markdown += [md_table(headers, table_rows), '']
            table = doc.add_table(rows=1, cols=len(headers)); table.style = 'Light Shading Accent 1'
            for cell, name in zip(table.rows[0].cells, headers): cell.text = name
            for row in table_rows:
                for cell, value in zip(table.add_row().cells, row): cell.text = str(value)
            picture = 'comparison_curves.png'
        elif title.startswith('7.'):
            headers = ['Nhóm', 'Số mẫu', 'AE phát hiện', 'AE bỏ sót', 'AE recall']
            markdown += [md_table(headers, group_rows), '']
            table = doc.add_table(rows=1, cols=5); table.style = 'Light Shading Accent 1'
            for cell, name in zip(table.rows[0].cells, headers): cell.text = name
            for row in group_rows:
                for cell, value in zip(table.add_row().cells, row): cell.text = str(value)
            picture = 'group_recall.png'
        elif title.startswith('8.'): picture = 'demo_counts.png'
        else: picture = None
        if picture:
            markdown += [f'![{title}](figures/{picture})', '']
            doc.add_picture(str(FIG / picture), width=DocInches(6.2))
    (OUT / 'final_report.md').write_text('\n'.join(markdown), encoding='utf-8')
    doc.save(OUT / 'final_report.docx')

    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    navy, teal, white = RGBColor(16, 35, 54), RGBColor(23, 161, 160), RGBColor(245, 248, 252)

    def slide(title, bullets=(), picture=None, note=''):
        page = prs.slides.add_slide(prs.slide_layouts[6])
        page.background.fill.solid(); page.background.fill.fore_color.rgb = navy
        box = page.shapes.add_textbox(Inches(.55), Inches(.35), Inches(12.2), Inches(.8))
        paragraph = box.text_frame.paragraphs[0]; paragraph.text = title
        paragraph.font.size = Pt(29); paragraph.font.bold = True; paragraph.font.color.rgb = white
        if picture:
            page.shapes.add_picture(str(FIG / picture), Inches(.65), Inches(1.45), width=Inches(12))
        else:
            box = page.shapes.add_textbox(Inches(.7), Inches(1.55), Inches(11.9), Inches(5))
            box.text_frame.word_wrap = True
            for index, text in enumerate(bullets):
                paragraph = box.text_frame.paragraphs[0] if index == 0 else box.text_frame.add_paragraph()
                paragraph.text = text; paragraph.font.size = Pt(25); paragraph.font.color.rgb = white
                paragraph.space_after = Pt(20)
        footer = page.shapes.add_textbox(Inches(.65), Inches(7.03), Inches(12), Inches(.3))
        paragraph = footer.text_frame.paragraphs[0]
        paragraph.text = f'NSL-KDD  /  AUTOENCODER & RISK SCORING                         {len(prs.slides):02d}'
        paragraph.font.size = Pt(11); paragraph.font.color.rgb = teal
        page.notes_slide.notes_text_frame.text = note
        return page

    slide('AUTOENCODER & NETWORK RISK', ['Phát hiện Normal / Attack trên NSL-KDD',
          'Risk Score 0–100 · Ba mức cảnh báo · Streamlit demo',
          'Kết quả thực nghiệm, giới hạn và hướng phát triển'], note='30 giây. Giới thiệu đề tài, không hứa hẹn phát hiện mọi tấn công.')
    slide('01  Bài toán và đầu ra', ['Học biểu diễn Normal, dùng reconstruction error phát hiện bất thường.',
          'So sánh RF, Isolation Forest và One-Class SVM trên cùng test.',
          'Điểm rủi ro là phân vị bất thường, không phải xác suất tấn công.'], note='40 giây. Phân biệt bất thường, nhãn tấn công và mức nguy hiểm thực tế.')
    slide('02  Dữ liệu và pipeline', ['KDDTrain+: 125.973 mẫu → train 100.778 / validation 25.195.',
          'KDDTest+: 22.544 mẫu; 41 đặc trưng → 77 sau mã hóa.',
          'One-hot + StandardScaler chỉ fit trên Normal train.',
          'Giữ nhãn/difficulty ngoài đầu vào; xử lý category chưa thấy.'], note='45 giây. Nêu seed 42, stratify và dữ liệu chưa scaling khi upload.')
    slide('03  Validation có ba vai trò', ['5.388 Normal: chọn epoch bằng early stopping.',
          '4.040 Normal: hiệu chuẩn phân vị và Risk Score.',
          '7.559 Normal/Attack: chọn latent và threshold.',
          'Không dùng test để chọn lại cấu hình/ngưỡng.'], note='45 giây. Thừa nhận test đã được quan sát ở giai đoạn trước; nhãn validation có dùng để chọn cấu hình.')
    slide('04  Thiết kế Autoencoder', ['Encoder: 77 → 64 → 32 → 16.',
          'Decoder: 16 → 32 → 64 → 77; đầu ra linear.',
          'Adam 0,001 · batch 256 · weight decay 1e-5.',
          'Normal-only · MSE · early stopping patience 12.'], note='45 giây. Giải thích không dùng sigmoid vì StandardScaler có giá trị âm và vượt 1.')
    slide('05  Risk Score và ba cảnh báo', ['Score = 100 × tỷ lệ Normal tham chiếu có MSE ≤ MSE mẫu.',
          'Low: MSE < 0,02376216 (P95).',
          'Medium: từ P95 đến dưới 0,10899014 (P99).',
          'High: từ P99 trở lên → Attack; giữ MSE đầy đủ để quyết định.'], note='45 giây. Điểm phân vị không phải xác suất; nhiều mẫu có thể cùng điểm 100.')
    slide('06  Latent 8 / 16 / 32 và overfitting', picture='optimization.png',
          note='60 giây. AP validation 0,972914 / 0,978826 / 0,978095; chọn 16. MSE nhỏ nhất không tự đảm bảo AP lớn nhất. Chỉ một seed, không kết luận ý nghĩa thống kê; early stopping giữ checkpoint tốt nhất.')
    page = slide('07  Bốn mô hình trên cùng KDDTest+', note='60 giây. RF có giám sát. AE AP cao nhất 0,9662; RF macro-F1 cao nhất 0,8069; OCSVM FPR thấp nhất. Không gọi một mô hình thắng mọi tiêu chí.')
    table = page.shapes.add_table(5, 6, Inches(.55), Inches(1.65), Inches(12.2), Inches(3.7)).table
    headers = ['Mô hình', 'Precision', 'Recall', 'Macro-F1', 'FPR', 'AP']
    for r, row in enumerate([headers] + [[name, f"{results[name]['precision_attack']:.2%}", f"{results[name]['recall_attack']:.2%}",
                                       f"{results[name]['macro_f1']:.4f}", f"{results[name]['false_positive_rate']:.2%}",
                                       f"{results[name]['pr_auc_average_precision']:.4f}"] for name in MODELS]):
        for c, text in enumerate(row):
            cell = table.cell(r, c); cell.text = text
            cell.fill.solid(); cell.fill.fore_color.rgb = teal if r == 0 else RGBColor(237, 244, 248)
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(19); paragraph.font.color.rgb = white if r == 0 else navy
    slide('08  Recall theo DoS / Probe / R2L / U2R', picture='group_recall.png',
          note='60 giây. AE R2L 17,16% là điểm yếu. RF tốt DoS/Probe nhưng kém R2L/U2R. U2R chỉ 67 mẫu; một mẫu tương đương 1,49 điểm phần trăm recall.')
    slide('09  Autoencoder: lỗi nào còn lại?', picture='ae_errors.png',
          note='45 giây. FN 5.527, FP 147. R2L bỏ sót 2.390. Không suy diễn nguyên nhân chỉ từ bảng tổng hợp; cần phân tích đặc trưng thêm.')
    slide('10  PR-AUC cao không thay thế threshold', picture='comparison_curves.png',
          note='45 giây. Đọc đồng thời AUC và recall/FPR tại threshold đã chọn. Mục tiêu FPR selection 1% không giữ nguyên trên test: AE 1,51%.')
    slide('11  Demo trực tiếp · khoảng 3 phút', ['Mở localhost:8501 → dữ liệu mẫu → phân tích.',
          'Upload KDDTest+.txt → schema hợp lệ → biểu đồ và mức cảnh báo.',
          'Lọc High → tải toàn bộ CSV theo thứ tự gốc.',
          'Thử CSV thiếu cột → thông báo lỗi; không chạy mô hình.'], note='Dùng demo_script.md. Không huấn luyện lại trong buổi trình bày. Nếu server lỗi, chuyển sang slide kết quả demo dự phòng.')
    slide('12  Demo dự phòng: kết quả đã kiểm chứng', picture='demo_counts.png',
          note='30 giây nếu cần thay demo trực tiếp. High 7.453, Medium 3.958, Low 11.133. Tỷ lệ High không phải độ chính xác hoặc tỷ lệ tấn công thật.')
    slide('13  Kiểm thử và giới hạn', ['21 unit/UI tests PASS; replay raw → preprocessing → 4 mô hình PASS.',
          'Đối chiếu đầy đủ 22.544 mẫu, nhãn, metrics, risk và alert.',
          'NSL-KDD cũ; một seed; ngân sách tuning các mô hình chưa ngang nhau.',
          'Không tuyên bố sẵn sàng phát hiện lưu lượng production.'], note='45 giây. Nói rõ audit cuối dùng estimator đã lưu, không huấn luyện lại. Chỉ ra pipeline_audit.json và requirements-lock.txt.')
    slide('14  Kết luận và câu hỏi', ['Hoàn thành pipeline, model, scoring, baseline và demo.',
          'AE: AP 0,9662 · recall 56,93% · FPR 1,51%.',
          'Ưu tiên tiếp: cải thiện R2L, nhiều seed, benchmark mới.',
          'Trao đổi: chọn ngưỡng thế nào khi chi phí bỏ sót khác báo nhầm?'], note='30 giây. Kết thúc bằng đánh đổi và hướng kiểm chứng tiếp theo; dùng phần hỏi đáp trong demo_script.md.')
    prs.save(OUT / 'presentation.pptx')
    return len(prs.slides)


def package():
    files = []
    for folder in ['src', 'scripts', 'tests', 'docs', 'notebooks', 'data', 'artifacts', 'reports', 'deliverables']:
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.zip') and path.name != 'manifest.json':
                files.append(path)
    for name in ['README.md', 'requirements.txt', 'requirements-lock.txt', '.gitignore', 'app.py', 'run_demo.cmd']:
        files.append(ROOT / name)
    files = sorted(set(files))
    manifest = {str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    archive = OUT / 'nsl_kdd_submission.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zip_file:
        for path in files + [OUT / 'manifest.json']:
            zip_file.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(archive) as zip_file:
        assert zip_file.testzip() is None
        for name, expected in manifest.items():
            assert hashlib.sha256(zip_file.read(name)).hexdigest() == expected
    print(f'PASS: ZIP CRC + {len(manifest)} file hashes; {archive.stat().st_size / 1024**2:.1f} MB')


if __name__ == '__main__':
    print(f'Built report and {build()} slides')
    package()
