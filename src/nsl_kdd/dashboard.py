"""CSV validation and inference shared by the Streamlit UI and its tests."""
import csv
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.inspect_nsl_kdd import FEATURES
from .autoencoder import load_detector, reconstruction_error
from .risk import alert_levels, risk_scores

CATEGORICAL = ('protocol_type', 'service', 'flag')
NUMERIC = [name for name in FEATURES if name not in CATEGORICAL]
MAX_BYTES = 20 * 1024 * 1024
MAX_ROWS = 100_000


def parse_csv(content: bytes, mode='auto'):
    if mode not in ('auto', 'header', 'no_header'):
        raise ValueError('Chế độ header không hợp lệ.')
    if not content or not content.strip():
        raise ValueError('File CSV rỗng.')
    if len(content) > MAX_BYTES:
        raise ValueError('File vượt giới hạn 20 MB.')
    try:
        decoded = content.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ValueError('Cần CSV mã hóa UTF-8, phân cách bằng dấu phẩy.') from exc
    try:
        reader = csv.reader(io.StringIO(decoded, newline=''), strict=True)
        rows = []
        for row in reader:
            if len(rows) >= MAX_ROWS + 1:
                raise ValueError('Tối đa 100.000 dòng dữ liệu mỗi lần.')
            rows.append([value.strip() for value in row])
    except csv.Error as exc:
        raise ValueError(f'CSV sai định dạng: {exc}') from exc
    if not rows or not rows[0]:
        raise ValueError('Không có dòng dữ liệu hợp lệ.')
    header = mode == 'header' or (mode == 'auto' and bool(set(rows[0]) & set(FEATURES)))
    if header:
        columns, body = rows[0], rows[1:]
        duplicates = [name for name, count in Counter(columns).items() if count > 1]
        missing = sorted(set(FEATURES) - set(columns))
        extra = sorted(set(columns) - set(FEATURES) - {'label', 'difficulty'})
        if duplicates or missing or extra:
            raise ValueError(f'Schema không khớp. Thiếu: {missing}; cột lạ: {extra}; cột trùng: {duplicates}.')
    else:
        if len(rows[0]) not in (41, 43):
            raise ValueError('CSV không header cần đúng 41 đặc trưng hoặc 43 cột NSL-KDD gốc. Dùng dấu phẩy để phân cách.')
        columns = [*FEATURES] + (['label', 'difficulty'] if len(rows[0]) == 43 else [])
        body = rows
    if not body:
        raise ValueError('File chỉ có header, chưa có dữ liệu.')
    if len(body) > MAX_ROWS:
        raise ValueError('Tối đa 100.000 dòng dữ liệu mỗi lần.')
    for index, row in enumerate(body):
        if len(row) != len(columns):
            raise ValueError(f'Dòng dữ liệu {index + 1} có {len(row)} cột; cần {len(columns)}.')
    frame = pd.DataFrame(body, columns=columns)
    for name in FEATURES:
        if frame[name].eq('').any():
            raise ValueError(f'Cột {name} có giá trị trống.')
    for name in NUMERIC:
        converted = pd.to_numeric(frame[name], errors='coerce')
        if not np.isfinite(converted.to_numpy(dtype=np.float64)).all():
            raise ValueError(f'Cột {name} cần là số hữu hạn, không có NaN/Inf hoặc chuỗi.')
        if (converted < 0).any():
            raise ValueError(f'Cột {name} không được âm; hãy upload dữ liệu NSL-KDD gốc, chưa scaling.')
        if name.endswith('_rate') and (converted > 1).any():
            raise ValueError(f'Cột {name} phải nằm trong [0, 1].')
        if not name.endswith('_rate') and (converted % 1 != 0).any():
            raise ValueError(f'Cột {name} cần số nguyên không âm.')
        frame[name] = converted
    for name in ['land', 'logged_in', 'root_shell', 'is_host_login', 'is_guest_login']:
        if not frame[name].isin([0, 1]).all():
            raise ValueError(f'Cột {name} chỉ nhận 0 hoặc 1.')
    if not frame.su_attempted.isin([0, 1, 2]).all():
        raise ValueError('Cột su_attempted chỉ nhận 0, 1 hoặc 2.')
    return frame, 'Có header' if header else 'Không header'


def load_bundle(directory: Path):
    model, pipeline, threshold = load_detector(directory)
    config = json.loads((directory / 'risk_config.json').read_text(encoding='utf-8'))
    if config['high_threshold'] != threshold:
        raise ValueError('Threshold mô hình và risk config không khớp.')
    reference = np.load(directory / 'normal_reference.npy', allow_pickle=False)
    # Validate reference and boundaries before accepting the bundle.
    risk_scores([], reference)
    alert_levels([], config['medium_threshold'], config['high_threshold'])
    return model, pipeline, config, reference


def score_frame(frame, bundle):
    model, pipeline, config, reference = bundle
    encoder = pipeline.named_steps['features'].named_transformers_['categorical']
    unknown = {name: int((~frame[name].isin(categories)).sum())
               for name, categories in zip(CATEGORICAL, encoder.categories_)}
    values = pipeline.transform(frame[list(FEATURES)]).astype(np.float32)
    errors = reconstruction_error(model, values)
    levels = alert_levels(errors, config['medium_threshold'], config['high_threshold'])
    result = pd.DataFrame({'row_id': np.arange(len(frame)), 'reconstruction_error': errors,
                           'risk_score': risk_scores(errors, reference), 'alert_level': levels,
                           'prediction': np.where(errors >= config['high_threshold'], 'Attack', 'Normal')})
    if 'label' in frame:
        result['original_label'] = frame.label.to_numpy()
    return result, unknown
