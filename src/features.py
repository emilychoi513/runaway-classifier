"""표 피처 전체: 기본 피처(fg_features.py) + 현재 행 파생 변수 + 센서별 과거 방향 피처.

- 기본 피처(센서, 조건·온도, 가스, 과거 방향 피처 10개)는 fg_features.py 를 그대로 쓴다.
- 추가 파생 변수(현재 행 차이·합, 센서별 과거 방향 피처)는 이 파일에서 만든다.
- 여기서 만든 컬럼을 모두 모델에 넣지는 않는다. 최종 모델이 쓰는 41개는 model.py의 BASE41에 적혀 있다.
- 과거 방향 피처는 같은 experiment_id 안에서 현재 행과 그 이전 행만 쓴다 (미래 행·실험 전체 통계 금지).
- 입력 금지 컬럼(fg_features.FORBIDDEN): sample_id, experiment_id, relative_time_s, image_path, label,
  indu_t, indu_t.1, meas_t1_plus_t2
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import fg_features as F  # noqa: E402

FORBIDDEN = F.FORBIDDEN

# ---------------------------------------------------------------------------
# 추가 파생 변수
# ---------------------------------------------------------------------------
# 현재 행 값만으로 계산 (row_features가 만드는 컬럼)
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
# meas_t1, meas_v, meas_t2, meas_p, thermal_t, 가스 합의 과거 방향 피처 (new_past_features가 만드는 컬럼)
NEW_PAST_COLS = PT1_COLS + PV_COLS + PT2_COLS + PP_COLS + PTH_COLS + PG_COLS


def row_features(frame: pd.DataFrame) -> pd.DataFrame:
    """현재 행 값만으로 계산하는 차이·합 피처(ROW_COLS)."""
    x = pd.DataFrame(F.to_num(frame, ['meas_t1', 'meas_t2', 'atmp_t', 'thermal_t']), columns=['t1', 't2', 'atmp', 'th'], index=frame.index)
    gas = F.to_num(frame, F.GAS)
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


def new_past_features(frame: pd.DataFrame) -> pd.DataFrame:
    """센서별 과거 방향 피처(NEW_PAST_COLS). 계산 규칙은 fg_features.past_features와 같다(정렬해서 계산, 입력 순서로 반환)."""
    n = len(frame)
    out = {c: np.full(n, np.nan) for c in NEW_PAST_COLS}
    if n == 0:
        return pd.DataFrame(out, index=frame.index)
    t_all = F.to_num(frame, ['relative_time_s'])[:, 0]
    if np.isnan(t_all).any():
        raise ValueError('relative_time_s에 숫자가 아닌 값이 있어 과거 피처를 계산할 수 없습니다.')
    sens = F.to_num(frame, ['meas_t1', 'meas_v', 'meas_t2', 'meas_p', 'thermal_t'])
    gas_sum = F.to_num(frame, F.GAS).sum(axis=1)
    exp = frame['experiment_id'].astype(str).to_numpy()
    order = np.lexsort((np.arange(n), t_all, exp))
    exp_sorted = exp[order]
    starts = np.flatnonzero(np.r_[True, exp_sorted[1:] != exp_sorted[:-1]])
    ends = np.r_[starts[1:], n]
    lag = F._lag_value
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
    """모델에 넣을 수 있는 표 피처 전체. 행 순서는 입력과 같다."""
    table = pd.concat([F.feature_table(frame), row_features(frame), new_past_features(frame)], axis=1)
    assert table.columns.is_unique and not (set(table.columns) & FORBIDDEN)
    return table


def self_check(frame: pd.DataFrame, n_experiments: int = 12, seed: int = 0) -> dict:
    """모든 과거 피처가 미래 행을 쓰지 않는지, 입력 행 순서에 무관한지 실제 데이터로 확인한다.

    학습·추론 경로에서는 부르지 않는다. 피처 계산식을 고친 뒤 직접 불러 확인하는 용도다.
    """
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
