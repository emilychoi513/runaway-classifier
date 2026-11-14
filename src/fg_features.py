"""F·G 단계 피처: 추가 컬럼(F)과 과거 방향 피처(G).

규칙
- F 피처는 행 하나의 값만 쓴다.
- G 피처는 같은 experiment_id 안에서 relative_time_s 기준으로 "현재 행과 그 이전 행"만 쓴다.
  미래 행과 실험 전체 통계(최종 최댓값, 평균 등)는 쓰지 않는다.
- 결측·무한값: 숫자로 바꿀 수 없는 값 → NaN, ±inf → NaN. NaN은 그대로 CatBoost(nan_mode=Min)에 넘긴다.
- 계산할 수 없는 과거 피처(이전 행 부족, 사건 미발생)는 NaN으로 둔다.
- 입력 행 순서와 무관하게 같은 값이 나오도록 내부에서 정렬해 계산하고, 결과는 입력 순서로 돌려준다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# D2가 쓰는 센서 5개 (팀 기준 구성과 같은 순서)
BASE = ['meas_v', 'meas_a', 'meas_t1', 'meas_t2', 'meas_p']

# F1: 조건·온도 컬럼 (현재 행 값 그대로, 숫자로 넣음)
F1_COLS = ['bat_class', 'bat_cc', 'soc_status', 'tr_method', 'thermal_t', 'atmp_t']

# F2: 가스 10종 + "10종 모두 0" 표시
GAS = ['meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf', 'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2']
F2_COLS = GAS + ['gas_all_zero']

# G1: 과거 방향 피처 6개
G1_COLS = [
    'past_t1_cummax',            # 지금까지의 meas_t1 최댓값
    'past_t1_drop_from_max',     # meas_t1 - 지금까지의 최댓값 (0 이하)
    'past_v_change_from_start',  # meas_v - 실험에서 처음 관측된 meas_v
    'past_v_cummin',             # 지금까지의 meas_v 최솟값
    'past_t1_change_30s',        # meas_t1 - (30초 이전 시점 중 가장 최근 행의 meas_t1)
    'past_v_change_5s',          # meas_v - (5초 이전 시점 중 가장 최근 행의 meas_v)
]

# G2: 사건 이후 경과 시간 2개 (사건이 아직 없으면 NaN)
G2_COLS = [
    'past_s_since_t1_45',        # meas_t1이 처음 45℃ 이상이 된 뒤 지난 초
    'past_s_since_v_drop',       # meas_v가 시작 대비 0.77V 이상 처음 떨어진 뒤 지난 초
]

# G3: 사건 발생 여부 2개 (경과 시간 없이 "일어났는지"만. 0 또는 1)
G3_COLS = [
    'past_t1_reached_45',        # 지금까지 meas_t1이 45℃ 이상이 된 적이 있는가
    'past_v_dropped',            # 지금까지 meas_v가 시작 대비 0.77V 이상 떨어진 적이 있는가
]

# 사건 기준값. Train 분석에서 정한 고정 상수이며 Validation·Test로 조정하지 않는다.
T1_EVENT_C = 45.0
V_DROP_EVENT_V = -0.77

# 계속 제외하는 컬럼 (모델 입력 금지)
FORBIDDEN = {'sample_id', 'experiment_id', 'relative_time_s', 'image_path', 'image_file', 'label',
             'indu_t', 'indu_t.1', 'meas_t1_plus_t2'}

GROUPS = {'base': BASE, 'F1': F1_COLS, 'F2': F2_COLS, 'G1': G1_COLS, 'G2': G2_COLS, 'G3': G3_COLS}


def to_num(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """컬럼을 float64 행렬로 만든다. 변환 실패·무한값은 NaN."""
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f'입력에 컬럼이 없습니다: {missing}')
    x = frame[cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
    x[~np.isfinite(x)] = np.nan
    return x


def row_features(frame: pd.DataFrame) -> pd.DataFrame:
    """행 하나만으로 계산하는 파생 피처."""
    gas = to_num(frame, GAS)
    return pd.DataFrame({'gas_all_zero': (gas == 0).all(axis=1).astype(np.float64)}, index=frame.index)


def _lag_value(t: np.ndarray, x: np.ndarray, seconds: float) -> np.ndarray:
    """각 행에 대해 시각이 (t - seconds) 이하인 행 중 가장 최근 행의 x. 없으면 NaN. t는 오름차순."""
    j = np.searchsorted(t, t - seconds, side='right') - 1
    out = np.full(len(t), np.nan)
    ok = j >= 0
    out[ok] = x[j[ok]]
    return out


def _first_true_time(t: np.ndarray, cond: np.ndarray) -> np.ndarray:
    """cond가 처음 참이 된 시각부터의 경과 초. 그 전 행은 NaN."""
    out = np.full(len(t), np.nan)
    hit = np.flatnonzero(cond)
    if len(hit):
        k = hit[0]
        out[k:] = t[k:] - t[k]
    return out


def past_features(frame: pd.DataFrame) -> pd.DataFrame:
    """G1·G2·G3 피처. 같은 experiment_id의 현재·이전 행만 사용한다."""
    n = len(frame)
    out = {c: np.full(n, np.nan) for c in G1_COLS + G2_COLS + G3_COLS}
    if n == 0:
        return pd.DataFrame(out, index=frame.index)
    t_all = to_num(frame, ['relative_time_s'])[:, 0]
    if np.isnan(t_all).any():
        raise ValueError('relative_time_s에 숫자가 아닌 값이 있어 과거 피처를 계산할 수 없습니다.')
    sens = to_num(frame, ['meas_t1', 'meas_v'])
    exp = frame['experiment_id'].astype(str).to_numpy()
    # 실험 → 시각 → 입력 순서로 정렬 (입력 순서는 같은 시각이 겹칠 때만 쓰임)
    order = np.lexsort((np.arange(n), t_all, exp))
    exp_sorted = exp[order]
    starts = np.flatnonzero(np.r_[True, exp_sorted[1:] != exp_sorted[:-1]])
    ends = np.r_[starts[1:], n]
    for s, e in zip(starts, ends):
        idx = order[s:e]
        t, t1, v = t_all[idx], sens[idx, 0], sens[idx, 1]
        t1_cummax = np.fmax.accumulate(t1)          # NaN은 건너뜀
        v_cummin = np.fmin.accumulate(v)
        finite_v = np.flatnonzero(np.isfinite(v))
        v_start = np.full(len(v), np.nan)
        if len(finite_v):
            v_start[finite_v[0]:] = v[finite_v[0]]   # 처음 관측된 전압. 그 이전 행은 NaN
        v_change = v - v_start
        out['past_t1_cummax'][idx] = t1_cummax
        out['past_t1_drop_from_max'][idx] = t1 - t1_cummax
        out['past_v_change_from_start'][idx] = v_change
        out['past_v_cummin'][idx] = v_cummin
        out['past_t1_change_30s'][idx] = t1 - _lag_value(t, t1, 30.0)
        out['past_v_change_5s'][idx] = v - _lag_value(t, v, 5.0)
        with np.errstate(invalid='ignore'):
            out['past_s_since_t1_45'][idx] = _first_true_time(t, t1_cummax >= T1_EVENT_C)
            out['past_s_since_v_drop'][idx] = _first_true_time(t, v_change <= V_DROP_EVENT_V)
            out['past_t1_reached_45'][idx] = np.maximum.accumulate(t1_cummax >= T1_EVENT_C).astype(np.float64)
            out['past_v_dropped'][idx] = np.maximum.accumulate(v_change <= V_DROP_EVENT_V).astype(np.float64)
    return pd.DataFrame(out, index=frame.index)


def feature_table(frame: pd.DataFrame) -> pd.DataFrame:
    """모델에 넣을 수 있는 모든 비이미지 피처(BASE, F1, F2, G1, G2, G3)를 한 표로 만든다. 행 순서는 입력과 같다."""
    raw_cols = BASE + F1_COLS + GAS
    raw = pd.DataFrame(to_num(frame, raw_cols), columns=raw_cols, index=frame.index)
    table = pd.concat([raw, row_features(frame), past_features(frame)], axis=1)
    assert not (set(table.columns) & FORBIDDEN)
    return table


def columns_for(groups: list[str]) -> list[str]:
    cols: list[str] = []
    for g in groups:
        cols += GROUPS[g]
    assert len(cols) == len(set(cols)) and not (set(cols) & FORBIDDEN)
    return cols


# ---------------------------------------------------------------------------
# 자체 점검: 미래 미사용, 행 순서 무관
# ---------------------------------------------------------------------------
def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return bool(np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True))


def self_check(frame: pd.DataFrame, n_experiments: int = 12, seed: int = 0) -> dict:
    """과거 피처가 미래 행을 쓰지 않는지, 입력 행 순서에 무관한지 실제 데이터로 확인한다."""
    rng = np.random.RandomState(seed)
    cols = G1_COLS + G2_COLS + G3_COLS
    exps = rng.choice(sorted(frame['experiment_id'].astype(str).unique()), size=min(n_experiments, frame['experiment_id'].nunique()), replace=False)
    sub = frame[frame['experiment_id'].astype(str).isin(exps)].reset_index(drop=True)
    full = past_features(sub)[cols]
    full.index = sub['sample_id']

    # 1) 미래 미사용: 각 실험을 임의 시각에서 잘라 계산해도 남은 행의 값이 같아야 한다
    cuts = 0
    for ex in exps:
        g = sub[sub['experiment_id'].astype(str) == ex]
        times = np.sort(g['relative_time_s'].to_numpy())
        for cut in rng.choice(times, size=min(5, len(times)), replace=False):
            part = g[g['relative_time_s'] <= cut].reset_index(drop=True)
            got = past_features(part)[cols]
            got.index = part['sample_id']
            assert _same(got, full.loc[part['sample_id']]), f'미래 행 참조 의심: {ex} cut={cut}'
            cuts += 1

    # 2) 행 순서 무관: 섞어서 계산해도 sample_id별 값이 같아야 한다
    shuffled = sub.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    got = past_features(shuffled)[cols]
    got.index = shuffled['sample_id']
    assert _same(got.loc[full.index], full), '행 순서에 따라 값이 달라짐'

    # 3) 시각 중복 없음 (있으면 같은 시각 안의 순서가 입력 순서에 좌우됨)
    dup = int(frame.duplicated(['experiment_id', 'relative_time_s']).sum())
    return {'experiments_checked': int(len(exps)), 'truncation_cuts_checked': cuts, 'order_invariant': True,
            'duplicate_time_rows': dup}
