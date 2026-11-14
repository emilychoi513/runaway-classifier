"""meas_t1 + meas_t2 최고점 피처 15개.

배경: Train·Validation에서 6단계 시작 시점은 (meas_t1 + meas_t2)가 실험 전체 최고점에 닿는 행과 85개 실험 중 83개에서 1행 이내로 일치한다
(meas_t1만으로는 47개). 이 합은 데이터의 meas_t1_plus_t2 컬럼과 같은 값이며(차이 0.005 이하), 여기서는 두 센서를 직접 더해 쓴다.
최고점은 지나고 나서야 알 수 있으므로, 이 파일은 "지금까지의 최고점에서 얼마나 내려왔는지"를 과거 방향으로만 계산한다.

규칙
- 같은 experiment_id 안에서 현재 행과 그 이전 행만 쓴다. 미래 행·실험 전체 통계는 쓰지 않는다.
- 창 길이는 모두 초 단위(relative_time_s 기준). 학습하는 값이 없다. 입력 행 순서와 무관하다.
"""
import numpy as np
import pandas as pd


def _lag(t, x, s):
    j = np.searchsorted(t, t - s, side='right') - 1
    out = np.full(len(t), np.nan); ok = j >= 0; out[ok] = x[j[ok]]
    return out


def _one(t, s):
    n = len(t); F = {}
    cm = np.fmax.accumulate(s)
    F['past_sum'] = s
    F['past_sum_cummax'] = cm
    F['past_sum_drop_from_max'] = s - cm
    F['past_sum_ratio_max'] = s / np.where(cm > 1e-6, cm, np.nan)
    hit = np.isfinite(s) & (s == cm); last = np.maximum.accumulate(np.where(hit, np.arange(n), -1))
    F['past_s_since_sum_max'] = np.where(last >= 0, t - t[np.maximum(last, 0)], np.nan)   # 최고점을 마지막으로 갱신한 뒤 지난 초
    ok = np.flatnonzero(np.isfinite(s))
    F['past_sum_rise_from_start'] = s - s[ok[0]] if len(ok) else np.full(n, np.nan)
    for w in (5, 10, 15, 30, 60):
        F[f'past_sum_change_{w}s'] = s - _lag(t, s, w)
    d5 = np.nan_to_num(s - _lag(t, s, 5), nan=-np.inf); m5 = np.maximum.accumulate(d5)
    F['past_sum_max_change_5s'] = np.where(np.isfinite(m5), m5, np.nan)                    # 지금까지 가장 빠른 5초 상승
    hit = np.isfinite(m5) & (d5 == m5); last = np.maximum.accumulate(np.where(hit, np.arange(n), -1))
    F['past_s_since_sum_max_rate'] = np.where(last >= 0, t - t[np.maximum(last, 0)], np.nan)
    for drop in (2.0, 10.0):                                                               # 최고점에서 2℃/10℃ 이상 내려온 뒤 지난 초
        c = np.flatnonzero((s - cm) <= -drop); o = np.full(n, np.nan)
        if len(c):
            o[c[0]:] = t[c[0]:] - t[c[0]]
        F[f'past_s_since_sum_drop_{int(drop)}'] = o
    return F


def s_features(frame: pd.DataFrame) -> pd.DataFrame:
    n = len(frame)
    num = lambda c: pd.to_numeric(frame[c], errors='coerce').to_numpy(float)  # noqa: E731
    t_all = num('relative_time_s'); s_all = num('meas_t1') + num('meas_t2')
    exp = frame['experiment_id'].astype(str).to_numpy()
    order = np.lexsort((np.arange(n), t_all, exp)); es = exp[order]
    starts = np.flatnonzero(np.r_[True, es[1:] != es[:-1]]); ends = np.r_[starts[1:], n]
    out = None
    for a, b in zip(starts, ends):
        idx = order[a:b]; F = _one(t_all[idx], s_all[idx])
        if out is None:
            out = {k: np.full(n, np.nan) for k in F}
        for k, v in F.items():
            out[k][idx] = v
    return pd.DataFrame(out, index=frame.index)


S_COLS = list(s_features(pd.DataFrame({'experiment_id': ['e'] * 3, 'relative_time_s': [0, 5, 10], 'meas_t1': [20.0, 21.0, 22.0], 'meas_t2': [20.0, 20.0, 20.0]})).columns)
