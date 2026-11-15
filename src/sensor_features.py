"""표 피처: 센서·조건·가스 값과 과거 방향 피처.

- 기본 피처(base_table)와 추가 피처(extra_row_features, extra_past_features)를 feature_table이 이어 붙인다.
- 최종 모델이 쓰는 41개는 model.py의 BASE41에 적혀 있다.
- 과거 방향 피처는 같은 experiment_id의 현재 행과 그 이전 행만 쓴다. 계산할 수 없으면 NaN으로 둔다.
- 입력 행 순서와 무관하게 같은 값이 나온다(내부에서 정렬해 계산하고 입력 순서로 돌려준다).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 기본 피처
# ---------------------------------------------------------------------------
# 센서 5개 (원값은 모델 입력에 넣지 않고 과거 방향 피처의 재료로 쓴다)
BASE = ['meas_v', 'meas_a', 'meas_t1', 'meas_t2', 'meas_p']

# F1: 조건·온도 컬럼 (현재 행 값 그대로, 숫자로 넣음)
F1_COLS = ['bat_class', 'bat_cc', 'soc_status', 'tr_method', 'thermal_t', 'atmp_t']

# F2: 가스 10종. "10종 모두 0" 표시(gas_all_zero)는 row_features에서 만든다
GAS = ['meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf', 'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2']

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

# 모델 입력으로 쓰지 않는 컬럼
FORBIDDEN = {'sample_id', 'experiment_id', 'relative_time_s', 'image_path', 'image_file', 'label',
             'indu_t', 'indu_t.1', 'meas_t1_plus_t2'}


# ---------------------------------------------------------------------------
# 추가 피처
# ---------------------------------------------------------------------------
# 현재 행 값만으로 계산 (extra_row_features가 만드는 컬럼)
ROW_COLS = [
    'd_t1_minus_t2',          # meas_t1 - meas_t2
    'd_t1_minus_atmp',        # meas_t1 - atmp_t
    'd_t2_minus_atmp',        # meas_t2 - atmp_t
    'd_thermal_minus_atmp',   # thermal_t - atmp_t
    'd_thermal_minus_t1',     # thermal_t - meas_t1
    'd_gas_sum',              # 가스 10종 합
]
# 과거 방향: 같은 실험의 현재 행과 그 이전 행만 사용
PT1_COLS = [
    'past_t1_rise_from_start',   # meas_t1 - 실험에서 처음 관측된 meas_t1
    'past_t1_change_10s',        # meas_t1 - 10초 이전 값
    'past_t1_change_60s',        # meas_t1 - 60초 이전 값
    'past_t1_change_120s',       # meas_t1 - 120초 이전 값
    'past_t1_max_change_30s',    # 지금까지 30초 변화량의 최댓값 (가장 빠른 온도 상승)
    'past_s_since_t1_max',       # meas_t1이 지금까지의 최댓값을 마지막으로 갱신한 뒤 지난 초
]
PV_COLS = [
    'past_v_ratio_start',        # meas_v / 실험에서 처음 관측된 meas_v
    'past_v_change_15s',         # meas_v - 15초 이전 값
    'past_v_change_30s',         # meas_v - 30초 이전 값
    'past_v_change_60s',         # meas_v - 60초 이전 값
    'past_v_min_change_5s',      # 지금까지 5초 변화량의 최솟값 (가장 급한 전압 하락)
    'past_s_since_v_min',        # meas_v가 지금까지의 최솟값을 마지막으로 갱신한 뒤 지난 초
]
PT2_COLS = ['past_t2_cummax', 'past_t2_drop_from_max', 'past_t2_change_30s']
PP_COLS = ['past_p_change_from_start', 'past_p_cummax', 'past_p_drop_from_max', 'past_p_change_30s']
PTH_COLS = ['past_thermal_cummax', 'past_thermal_drop_from_max', 'past_thermal_change_30s']
PG_COLS = ['past_gas_sum_cummax', 'past_gas_sum_change_30s']
# meas_t1, meas_v, meas_t2, meas_p, thermal_t, 가스 합의 과거 방향 피처 (extra_past_features가 만드는 컬럼)
NEW_PAST_COLS = PT1_COLS + PV_COLS + PT2_COLS + PP_COLS + PTH_COLS + PG_COLS


# ---------------------------------------------------------------------------
# 기본 피처 계산
# ---------------------------------------------------------------------------
def to_num(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """컬럼을 float64 행렬로 만든다. 변환 실패·무한값은 NaN."""
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f'입력에 컬럼이 없습니다: {missing}')
    x = frame[cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
    x[~np.isfinite(x)] = np.nan
    return x


def base_row_features(frame: pd.DataFrame) -> pd.DataFrame:
    """기본 현재 행 피처: 가스 10종이 모두 0인지(gas_all_zero)."""
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


def base_past_features(frame: pd.DataFrame) -> pd.DataFrame:
    """기본 과거 방향 피처 10개(G1·G2·G3). 같은 experiment_id의 현재·이전 행만 사용한다."""
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


def base_table(frame: pd.DataFrame) -> pd.DataFrame:
    """기본 피처 표: 센서·조건·온도·가스 원값 + gas_all_zero + 기본 과거 방향 피처 10개. 행 순서는 입력과 같다."""
    raw_cols = BASE + F1_COLS + GAS
    raw = pd.DataFrame(to_num(frame, raw_cols), columns=raw_cols, index=frame.index)
    table = pd.concat([raw, base_row_features(frame), base_past_features(frame)], axis=1)
    assert not (set(table.columns) & FORBIDDEN)
    return table



# ---------------------------------------------------------------------------
# 추가 피처 계산과 전체 표
# ---------------------------------------------------------------------------
def extra_row_features(frame: pd.DataFrame) -> pd.DataFrame:
    """추가 현재 행 피처: 센서 값의 차이·합(ROW_COLS)."""
    x = pd.DataFrame(to_num(frame, ['meas_t1', 'meas_t2', 'atmp_t', 'thermal_t']), columns=['t1', 't2', 'atmp', 'th'], index=frame.index)
    gas = to_num(frame, GAS)
    return pd.DataFrame({
        'd_t1_minus_t2': x.t1 - x.t2,
        'd_t1_minus_atmp': x.t1 - x.atmp,
        'd_t2_minus_atmp': x.t2 - x.atmp,
        'd_thermal_minus_atmp': x.th - x.atmp,
        'd_thermal_minus_t1': x.th - x.t1,
        'd_gas_sum': gas.sum(axis=1),          # 하나라도 NaN이면 NaN
    }, index=frame.index)


def _first_value(x: np.ndarray) -> np.ndarray:
    """처음 관측된(유한한) 값. 그 이전 행은 NaN."""
    out = np.full(len(x), np.nan)
    ok = np.flatnonzero(np.isfinite(x))
    if len(ok):
        out[ok[0]:] = x[ok[0]]
    return out


def _since_last_record(t: np.ndarray, x: np.ndarray, running: np.ndarray) -> np.ndarray:
    """x가 누적 최댓값(또는 최솟값) running을 마지막으로 갱신한 시각부터 지난 초."""
    hit = np.isfinite(x) & (x == running)
    last = np.maximum.accumulate(np.where(hit, np.arange(len(x)), -1))
    out = np.full(len(x), np.nan)
    ok = last >= 0
    out[ok] = t[ok] - t[last[ok]]
    return out


def extra_past_features(frame: pd.DataFrame) -> pd.DataFrame:
    """센서별 추가 과거 방향 피처(NEW_PAST_COLS). 계산 규칙은 base_past_features와 같다(정렬해서 계산, 입력 순서로 반환)."""
    n = len(frame)
    out = {c: np.full(n, np.nan) for c in NEW_PAST_COLS}
    if n == 0:
        return pd.DataFrame(out, index=frame.index)
    t_all = to_num(frame, ['relative_time_s'])[:, 0]
    if np.isnan(t_all).any():
        raise ValueError('relative_time_s에 숫자가 아닌 값이 있어 과거 피처를 계산할 수 없습니다.')
    sens = to_num(frame, ['meas_t1', 'meas_v', 'meas_t2', 'meas_p', 'thermal_t'])
    gas_sum = to_num(frame, GAS).sum(axis=1)
    exp = frame['experiment_id'].astype(str).to_numpy()
    order = np.lexsort((np.arange(n), t_all, exp))
    exp_sorted = exp[order]
    starts = np.flatnonzero(np.r_[True, exp_sorted[1:] != exp_sorted[:-1]])
    ends = np.r_[starts[1:], n]
    lag = _lag_value
    for s, e in zip(starts, ends):
        idx = order[s:e]
        t = t_all[idx]
        t1, v, t2, p, th = (sens[idx, j] for j in range(5))
        g = gas_sum[idx]
        t1_max, v_min = np.fmax.accumulate(t1), np.fmin.accumulate(v)
        out['past_t1_rise_from_start'][idx] = t1 - _first_value(t1)
        for sec in (10, 60, 120):
            out[f'past_t1_change_{sec}s'][idx] = t1 - lag(t, t1, float(sec))
        out['past_t1_max_change_30s'][idx] = np.fmax.accumulate(t1 - lag(t, t1, 30.0))
        out['past_s_since_t1_max'][idx] = _since_last_record(t, t1, t1_max)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = v / _first_value(v)
        ratio[~np.isfinite(ratio)] = np.nan
        out['past_v_ratio_start'][idx] = ratio
        for sec in (15, 30, 60):
            out[f'past_v_change_{sec}s'][idx] = v - lag(t, v, float(sec))
        out['past_v_min_change_5s'][idx] = np.fmin.accumulate(v - lag(t, v, 5.0))
        out['past_s_since_v_min'][idx] = _since_last_record(t, v, v_min)
        for name, x in (('t2', t2), ('p', p), ('thermal', th)):
            cmax = np.fmax.accumulate(x)
            out[f'past_{name}_cummax'][idx] = cmax
            out[f'past_{name}_drop_from_max'][idx] = x - cmax
            out[f'past_{name}_change_30s'][idx] = x - lag(t, x, 30.0)
        out['past_p_change_from_start'][idx] = p - _first_value(p)
        out['past_gas_sum_cummax'][idx] = np.fmax.accumulate(g)
        out['past_gas_sum_change_30s'][idx] = g - lag(t, g, 30.0)
    return pd.DataFrame(out, index=frame.index)


def feature_table(frame: pd.DataFrame) -> pd.DataFrame:
    """표 피처 전체(기본 피처 + 추가 피처). 행 순서는 입력과 같다."""
    table = pd.concat([base_table(frame), extra_row_features(frame), extra_past_features(frame)], axis=1)
    assert table.columns.is_unique and not (set(table.columns) & FORBIDDEN)
    return table


def self_check(frame: pd.DataFrame, n_experiments: int = 12, seed: int = 0) -> dict:
    """과거 피처가 미래 행을 쓰지 않는지, 입력 행 순서에 무관한지 확인한다(학습·추론에서는 부르지 않는다)."""
    rng = np.random.RandomState(seed)
    cols = [c for c in feature_table(frame.head(50)).columns if c.startswith('past_')]
    exps = rng.choice(sorted(frame['experiment_id'].astype(str).unique()), size=min(n_experiments, frame['experiment_id'].nunique()), replace=False)
    sub = frame[frame['experiment_id'].astype(str).isin(exps)].reset_index(drop=True)
    full = feature_table(sub)[cols]
    full.index = sub['sample_id']
    same = lambda a, b: bool(np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True))  # noqa: E731
    cuts = 0
    for ex in exps:
        g = sub[sub['experiment_id'].astype(str) == ex]
        times = np.sort(g['relative_time_s'].to_numpy())
        for cut in rng.choice(times, size=min(5, len(times)), replace=False):
            part = g[g['relative_time_s'] <= cut].reset_index(drop=True)
            got = feature_table(part)[cols]
            got.index = part['sample_id']
            assert same(got, full.loc[part['sample_id']]), f'미래 행 참조 의심: {ex} cut={cut}'
            cuts += 1
    shuffled = sub.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    got = feature_table(shuffled)[cols]
    got.index = shuffled['sample_id']
    assert same(got.loc[full.index], full), '행 순서에 따라 값이 달라짐'
    return {'past_features_checked': len(cols), 'experiments_checked': int(len(exps)), 'truncation_cuts_checked': cuts,
            'order_invariant': True}
