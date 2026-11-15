"""열화상 이미지에서 직접 뽑는 피처 (DINOv2·이미지 분류기 없이, 정해진 계산식만 사용).

열화상은 색으로 온도 수준을 나타낸다. 색 순서(차가움 → 뜨거움)는
  검정 → 자주 → 파랑 → 청록 → 초록 → 노랑 → 빨강 → 흰색
이고, 픽셀마다 이 순서에서의 위치를 0~3 값(level)으로 바꾼 뒤 통계를 낸다.
  0~1  검정 → 밝은 자주   (색상 약 300°, 밝기가 올라감)
  1~2  자주 → 빨강         (색상 300° → 0°)
  2~3  빨강 → 흰색         (채도가 내려감)

주의: 색 기준(어느 온도가 어느 색인지)은 프레임·실험마다 다를 수 있어 level은 절대 온도가 아니다.
이미지 크기도 실험마다 달라서, 모든 피처는 비율·평균으로만 만든다(크기 자체는 피처로 쓰지 않음).

- 현재 행 피처(IR): 이미지 한 장만으로 계산.
- 과거 방향 피처(IP): 같은 experiment_id의 현재 행과 그 이전 행의 IR 값만 사용.
- 학습하는 부분이 없다. 입력 데이터로 통계를 추정하지 않는다.
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import sensor_features as SF  # noqa: E402

VERSION = 'v1'
HOT, WHITE = 1.8, 2.2      # level 1.8 = 노랑(색상 60°) 이상, 2.2 = 흰색에 가까움(채도 0.8 미만)

IR_COLS = [
    'img_level_mean', 'img_level_std', 'img_level_p10', 'img_level_p50', 'img_level_p90', 'img_level_p99',
    'img_frac_dark',      # level < 0.5 (어두운 자주·검정)
    'img_frac_cold',      # level < 1.0 (자주 계열)
    'img_frac_cool',      # 1.0 ~ 1.4 (파랑·청록)
    'img_frac_mid',       # 1.4 ~ 1.8 (청록·초록·노랑)
    'img_frac_hot',       # level >= 1.8 (노랑·빨강·흰색)
    'img_frac_white',     # level >= 2.2 (흰색에 가까움)
    'img_border_level',   # 가장자리 10% 띠의 평균 level
    'img_center_level',   # 가운데 50% 영역의 평균 level
    'img_center_minus_border',
    'img_hot_bbox_frac',      # 뜨거운 픽셀을 감싸는 사각형이 화면에서 차지하는 비율
    'img_hot_border_frac',    # 가장자리 띠에서 뜨거운 픽셀의 비율
    'img_hot_components',     # 뜨거운 덩어리 수
    'img_hot_largest_frac',   # 가장 큰 뜨거운 덩어리가 화면에서 차지하는 비율
    'img_grad_mean',          # 이웃 픽셀 간 level 차이의 평균 (경계가 뚜렷한 정도)
]
IP_COLS = [
    'past_img_level_mean_cummax', 'past_img_level_mean_drop_from_max', 'past_img_level_mean_change_30s',
    'past_img_hot_frac_cummax', 'past_img_hot_frac_change_30s',
    'past_img_white_frac_cummax',
    'past_img_border_level_cummax', 'past_img_border_level_change_30s',
]
# 모델 A가 쓰는 이미지 피처 4개. 나머지 IR·IP 피처는 계산만 하고 모델에 넣지 않는다
IC_COLS = ['img_level_mean', 'img_frac_hot', 'past_img_level_mean_cummax', 'past_img_hot_frac_cummax']


def level_map(rgb: np.ndarray) -> np.ndarray:
    """RGB(uint8, H×W×3) → 색 순서에서의 위치 level(float32, H×W, 0~3)."""
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(axis=-1), x.min(axis=-1)
    d = mx - mn
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(mx > 0, d / mx, 0.0)
        h = np.where(mx == r, ((g - b) / d) % 6.0, np.where(mx == g, (b - r) / d + 2.0, (r - g) / d + 4.0)) * 60.0
    h = np.where(d > 0, h, 0.0)
    h = np.where(h >= 330.0, 0.0, h)                    # 빨강 쪽으로 넘어간 색상은 빨강으로 취급
    level = 1.0 + (300.0 - np.clip(h, 0.0, 300.0)) / 300.0     # 자주(300°) → 빨강(0°)
    level = np.where(s < 0.97, 2.0 + (1.0 - s), level)         # 빨강 → 흰색
    level = np.where((h >= 297.0) | (mx < 0.15), mx, level)    # 자주 계열과 아주 어두운 픽셀은 밝기로
    return level.astype(np.float32)


def frame_features(path) -> list[float]:
    """이미지 한 장의 IR 피처. 읽지 못하면 모두 NaN."""
    try:
        with Image.open(path) as im:
            lv = level_map(np.asarray(im.convert('RGB')))
    except Exception:
        return [float('nan')] * len(IR_COLS)
    hgt, wid = lv.shape
    n = lv.size
    bh, bw = max(1, int(round(hgt * 0.1))), max(1, int(round(wid * 0.1)))
    border = np.ones((hgt, wid), dtype=bool)
    border[bh:hgt - bh, bw:wid - bw] = False
    center = lv[hgt // 4:hgt - hgt // 4, wid // 4:wid - wid // 4]
    hot = lv >= HOT
    p10, p50, p90, p99 = np.percentile(lv, [10, 50, 90, 99])
    if hot.any():
        ys, xs = np.nonzero(hot)
        bbox = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1) / n
        lab, k = ndimage.label(hot)
        largest = float(np.bincount(lab.ravel())[1:].max()) / n
    else:
        bbox, k, largest = 0.0, 0, 0.0
    grad = (np.abs(np.diff(lv, axis=0)).mean() if hgt > 1 else 0.0) + (np.abs(np.diff(lv, axis=1)).mean() if wid > 1 else 0.0)
    border_level, center_level = float(lv[border].mean()), float(center.mean())
    return [float(lv.mean()), float(lv.std()), float(p10), float(p50), float(p90), float(p99),
            float((lv < 0.5).mean()), float((lv < 1.0).mean()), float(((lv >= 1.0) & (lv < 1.4)).mean()),
            float(((lv >= 1.4) & (lv < HOT)).mean()), float(hot.mean()), float((lv >= WHITE).mean()),
            border_level, center_level, center_level - border_level,
            float(bbox), float(hot[border].mean()), float(k), largest, float(grad)]


def resolve_image_paths(image_root, image_paths) -> list[str]:
    """CSV의 image_path(예: images/S_xxx.png)를 실제 파일 경로로 바꾼다.

    기본은 image_root / image_path 이다. image_root에는 images/ 폴더의 상위 폴더(예: .../test)를 준다(대회 점검 스크립트와 같은 방식).
    그 경로에 파일이 없으면 image_root가 이미지 폴더 자체(예: .../test/images)이거나 그 위 폴더(예: .../battery_data_light)인
    경우로 보고 image_root / 파일 이름, image_root / 하위 폴더 / image_path 순서로 찾는다. 어느 쪽이든 같은 파일을 읽으므로 예측도 같다.
    이미지를 하나도 찾지 못하면 이미지 피처가 모두 빈 값인 채로 조용히 예측하지 않도록 오류를 낸다.
    """
    root = Path(image_root)
    subdirs = None
    out, n_missing = [], 0
    for p in image_paths:
        rel = Path(p)
        q = root / rel
        if not q.is_file():
            if subdirs is None:
                subdirs = sorted(d for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
            q = next((c for c in [root / rel.name] + [d / rel for d in subdirs] if c.is_file()), q)
            n_missing += not q.is_file()
        out.append(str(q))
    if out and n_missing == len(out):
        raise FileNotFoundError(f'--image_root {image_root} 아래에서 이미지를 하나도 찾지 못했습니다. '
                                f'image_path(예: {image_paths[0]})의 images/ 폴더가 들어 있는 폴더(예: .../test)를 주세요.')
    if n_missing:
        print(f'[경고] 이미지 {n_missing}/{len(out)}장을 찾지 못해 그 행의 이미지 피처는 빈 값으로 둡니다.', file=sys.stderr)
    return out


def row_table(frame: pd.DataFrame, image_root, cache_file=None, workers: int = 8) -> pd.DataFrame:
    """IR 피처 표. 행 순서는 입력과 같다. cache_file이 있으면 sample_id가 같을 때 그대로 읽는다(학습용. test.py는 cache_file을 주지 않는다)."""
    sid = frame['sample_id'].to_numpy(str)
    if cache_file is not None and Path(cache_file).exists():
        z = np.load(cache_file, allow_pickle=False)
        if str(z['version']) == VERSION and len(z['sample_id']) == len(sid) and (z['sample_id'] == sid).all():
            return pd.DataFrame(z['x'], columns=IR_COLS, index=frame.index)
    paths = resolve_image_paths(image_root, frame['image_path'].astype(str).tolist())
    if workers > 1 and len(paths) > 2000:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(frame_features, paths, chunksize=256))
    else:
        rows = [frame_features(p) for p in paths]
    x = np.asarray(rows, dtype=np.float64).reshape(len(paths), len(IR_COLS))
    if cache_file is not None:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, x=x, sample_id=sid, version=VERSION)
    return pd.DataFrame(x, columns=IR_COLS, index=frame.index)


def past_table(frame: pd.DataFrame, ir: pd.DataFrame) -> pd.DataFrame:
    """IP 피처 표. 같은 실험의 현재·이전 행만 사용 (정렬해서 계산, 입력 순서로 반환)."""
    n = len(frame)
    out = {c: np.full(n, np.nan) for c in IP_COLS}
    if n == 0:
        return pd.DataFrame(out, index=frame.index)
    t_all = SF.to_num(frame, ['relative_time_s'])[:, 0]
    if np.isnan(t_all).any():
        raise ValueError('relative_time_s에 숫자가 아닌 값이 있어 과거 피처를 계산할 수 없습니다.')
    src = ir[['img_level_mean', 'img_frac_hot', 'img_frac_white', 'img_border_level']].to_numpy(dtype=np.float64)
    exp = frame['experiment_id'].astype(str).to_numpy()
    order = np.lexsort((np.arange(n), t_all, exp))
    exp_sorted = exp[order]
    starts = np.flatnonzero(np.r_[True, exp_sorted[1:] != exp_sorted[:-1]])
    ends = np.r_[starts[1:], n]
    lag = SF._lag_value
    for s, e in zip(starts, ends):
        idx = order[s:e]
        t = t_all[idx]
        lm, hf, wf, bl = (src[idx, j] for j in range(4))
        lm_max = np.fmax.accumulate(lm)
        out['past_img_level_mean_cummax'][idx] = lm_max
        out['past_img_level_mean_drop_from_max'][idx] = lm - lm_max
        out['past_img_level_mean_change_30s'][idx] = lm - lag(t, lm, 30.0)
        out['past_img_hot_frac_cummax'][idx] = np.fmax.accumulate(hf)
        out['past_img_hot_frac_change_30s'][idx] = hf - lag(t, hf, 30.0)
        out['past_img_white_frac_cummax'][idx] = np.fmax.accumulate(wf)
        out['past_img_border_level_cummax'][idx] = np.fmax.accumulate(bl)
        out['past_img_border_level_change_30s'][idx] = bl - lag(t, bl, 30.0)
    return pd.DataFrame(out, index=frame.index)


def image_table(frame: pd.DataFrame, image_root, cache_file=None, workers: int = 8) -> pd.DataFrame:
    """IR + IP 피처 표."""
    ir = row_table(frame, image_root, cache_file, workers)
    return pd.concat([ir, past_table(frame, ir)], axis=1)


def full_table(frame: pd.DataFrame, image_root, cache_file=None, workers: int = 8) -> pd.DataFrame:
    """표 피처(sensor_features.feature_table) + 이미지 피처."""
    table = pd.concat([SF.feature_table(frame), image_table(frame, image_root, cache_file, workers)], axis=1)
    assert table.columns.is_unique and not (set(table.columns) & SF.FORBIDDEN)
    return table
