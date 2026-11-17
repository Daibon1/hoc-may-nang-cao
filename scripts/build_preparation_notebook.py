"""Generate the editable preparation notebook using only the standard library."""
import json
from pathlib import Path

cells = []


def cell(kind, source):
    entry = {"id": f"cell-{len(cells):02d}", "cell_type": kind, "metadata": {}, "source": source.strip().splitlines(keepends=True)}
    if kind == "code":
        entry.update(execution_count=None, outputs=[])
    cells.append(entry)


cell("markdown", """
# NSL-KDD: EDA và preprocessing cho Autoencoder
Mục tiêu: kiểm tra dữ liệu, phân tích phân phối, chia validation và lưu pipeline.
Không huấn luyện mô hình hoặc chọn ngưỡng trên test ở giai đoạn này.

Nguồn: mirror Jehuty4949/NSL_KDD, commit `27bbbdf5c5cccf9970b8b6037a0f19dc09c5a0a9`.
Tham khảo: Tavallaee et al., *A Detailed Analysis of the KDD CUP 99 Data Set*, CISDA 2009.
Hai file raw được kiểm tra Git blob SHA-1 trước khi dùng; chỉ tải nếu chưa có.
""")
cell("code", """
from pathlib import Path
import sys
ROOT = Path.cwd().resolve()
if not (ROOT / 'scripts/prepare_nsl_kdd.py').exists():
    ROOT = ROOT.parent
assert (ROOT / 'scripts/prepare_nsl_kdd.py').exists(), 'Mở notebook từ thư mục project hoặc notebooks'
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from prepare_nsl_kdd import prepare, FEATURES, NUMERIC, CATEGORICAL
train, test, pipeline, arrays, metadata = prepare(ROOT)
pd.DataFrame(metadata['shapes'].items(), columns=['array', 'shape'])
""")
cell("markdown", """
## Kiểm tra chất lượng và phân bố nhãn
41 đặc trưng gồm 38 số và 3 categorical. Nhãn và difficulty luôn bị loại khỏi đầu vào.
Row ID trong file NPZ là chỉ số dòng bắt đầu từ 0 trong file raw tương ứng.
""")
cell("code", """
display(train.head())
quality = pd.DataFrame({name: {
    'rows': len(df), 'missing': int(df.isna().sum().sum()),
    'duplicate_records': int(df.duplicated(subset=[*FEATURES, 'label', 'difficulty']).sum()),
    'duplicate_features': int(df.duplicated(subset=list(FEATURES)).sum()),
    'attack_ratio': df.is_attack.mean(),
} for name, df in [('train', train), ('test', test)]}).T
display(quality)
counts = pd.DataFrame({'train': train.group.value_counts(), 'test': test.group.value_counts()}).fillna(0).astype(int)
display(counts)
counts.plot.bar(figsize=(9, 4), title='Phân bố normal và nhóm tấn công')
plt.ylabel('Số mẫu'); plt.xticks(rotation=0); plt.tight_layout(); plt.show()
display(train[list(NUMERIC)].describe().T)
""")
cell("code", """
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for col, ax in zip(CATEGORICAL, axes):
    train[col].value_counts().head(12).plot.bar(ax=ax, title=col)
plt.tight_layout(); plt.show()
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for col, ax in zip(['duration', 'src_bytes', 'dst_bytes'], axes):
    for label, frame in train.groupby('is_attack'):
        ax.hist(np.log1p(frame[col]), bins=50, alpha=0.5, label='attack' if label else 'normal')
    ax.set_title('log1p(' + col + ')'); ax.legend()
plt.tight_layout(); plt.show()
display(pd.DataFrame({
    'constant_train': train[list(NUMERIC)].nunique().eq(1),
    'min': train[list(NUMERIC)].min(), 'max': train[list(NUMERIC)].max(),
}))
print('Nhãn chỉ có trong test:', sorted(set(test.label) - set(train.label)))
""")
cell("markdown", """
## Quy tắc preprocessing
- Tách 80/20 từ KDDTrain+, stratify theo nhãn nhị phân, seed=42; giữ nguyên KDDTest+.
- Fit StandardScaler và OneHotEncoder chỉ trên normal của phần train.
- Category lạ tạo block one-hot bằng 0; thống kê bên dưới giúp nhận biết giới hạn này.
- StandardScaler giữ giá trị ngoài miền train, không clip. Đầu ra AE sau này nên dùng linear,
  không dùng sigmoid. Các biến có đuôi dài có thể chi phối MSE; log1p chỉ dùng cho EDA ở đây.
- Giữ đặc trưng hằng để schema ổn định; không tự loại trùng vì đang giữ giao thức benchmark.
  EDA test chỉ mô tả; không dùng để chọn biến, scaler hoặc ngưỡng.
- Dùng X_train_normal để huấn luyện AE; X_validation_normal cho early stopping/ngưỡng.
  Các mảng train đầy đủ được giữ để baseline có giám sát có thể dùng cùng phép chia.
""")
cell("code", """
encoder = pipeline.named_steps['features'].named_transformers_['categorical']
unknown = []
for split in ['train', 'validation', 'test']:
    frame = train.loc[arrays['row_' + split]] if split != 'test' else test
    for col, categories in zip(CATEGORICAL, encoder.categories_):
        unknown.append({'split': split, 'feature': col,
                        'unknown_rows': int((~frame[col].isin(categories)).sum())})
display(pd.DataFrame(unknown))
assert set(arrays['row_train']).isdisjoint(arrays['row_validation'])
assert len(arrays['row_train']) + len(arrays['row_validation']) == len(train)
assert len(arrays['X_test']) == len(test)
assert arrays['X_train_normal'].shape[0] == (arrays['y_train'] == 0).sum()
for key, value in arrays.items():
    if key.startswith('X_'):
        assert value.dtype == np.float32 and np.isfinite(value).all()
normal = train.loc[arrays['row_train']].query('is_attack == 0')
np.testing.assert_allclose(
    pipeline.named_steps['features'].named_transformers_['numeric'].mean_,
    normal[NUMERIC].mean().to_numpy(),
)
print('Kiểm tra chia tập, normal-only fit và ma trận hữu hạn: PASS')
print('Số chiều sau encode:', len(metadata['features']))
""")
cell("code", """
import joblib
saved = joblib.load(ROOT / 'artifacts/preprocessing/pipeline.joblib')
unseen = test.loc[:0, list(FEATURES)].copy()
unseen['service'] = '__unseen_service__'
assert np.isfinite(saved.transform(unseen)).all()
np.testing.assert_allclose(saved.transform(test.loc[:19, list(FEATURES)]),
                           arrays['X_test'][:20], rtol=1e-6, atol=1e-6)
with np.load(ROOT / 'data/processed/nsl_kdd.npz', allow_pickle=False) as stored:
    np.testing.assert_array_equal(stored['X_train_normal'], arrays['X_train_normal'])
print('Lưu/nạp pipeline và dữ liệu: PASS')
display(pd.DataFrame({'feature': metadata['features']}))
""")
cell("markdown", """
## Đầu ra và bước tiếp theo
Pipeline: `../artifacts/preprocessing/pipeline.joblib`; metadata: cùng thư mục.
Dữ liệu: `../data/processed/nsl_kdd.npz`, gồm ma trận float32, nhãn, nhóm và row ID.
Chỉ nạp joblib từ nguồn tin cậy. Phiên bản thư viện được ghi trong metadata và requirements-lock.txt.
NSL-KDD là benchmark cũ; mất cân bằng và nhãn tấn công mới trong test cần được tính đến
khi báo cáo kết quả. Chưa có kết luận về khả năng phát hiện hoặc điểm rủi ro trước khi huấn luyện.
""")
path = Path(__file__).resolve().parents[1] / 'notebooks/01_eda_preprocessing.ipynb'
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps({'cells': cells, 'metadata': {
    'kernelspec': {'display_name': 'Python (NSL-KDD)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.14'}}, 'nbformat': 4, 'nbformat_minor': 5},
    ensure_ascii=False, indent=1), encoding='utf-8')
print(path)
