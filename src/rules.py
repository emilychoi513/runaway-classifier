"""모델 예측 뒤에 적용하는 단계 기준 규칙. 학습하는 값은 없고 기준값은 아래 상수뿐이다.

모든 규칙은 같은 experiment_id의 현재 행과 그 이전 행만 쓴다. 근거와 검증 결과는 README.md에 있다.

- 규칙 1 (모든 실험): 전압 급락이 있었으면 4단계 이상. 3단계 이하 예측만 4단계로 올린다.
- 규칙 2 (280Ah 과충전): 온도 급상승 전에는 5·6단계가 아니고, 급상승 뒤에는 4단계가 아니다.
- 6단계 제약 (모든 실험): meas_t1 + meas_t2가 이전 최댓값 이상이면 6단계 예측을 5단계로 내린다.
- 4단계 최소 지속 제약 (6Ah 이하 전지): 4단계가 시작된 지 25초가 안 된 행의 5·6단계 예측을 4단계로 내린다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

T1_GATE = 50.0     # ℃. 전압 급락·온도 급상승은 meas_t1이 이 온도를 넘은 뒤부터 센다
V_DROP = 0.21      # 한 행 사이 전압 하락 / 처음 전압
V_LOW = 0.50       # 직전 행 전압 / 처음 전압 이 이 값 이하면 '전압이 낮아진 상태'
T1_RATE = 1.0      # ℃/초
STAGE4_MIN_S = 25.0  # 초. 소형 전지에서 4단계가 시작된 뒤 이 시간이 지나기 전에는 5단계 이상이 아님
SMALL_CC = 6.0     # Ah. 이 용량 이하를 소형 전지로 본다 (4.8Ah, 6Ah)
VERSION = 'v3'


def events(t: np.ndarray, t1: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """시간순으로 정렬된 한 실험. 각 행까지(현재 행 포함) 전압 급락 / 온도 급상승이 있었는지."""
    n = len(t)
    reached = np.fmax.accumulate(t1) >= T1_GATE
    ok = np.flatnonzero(np.isfinite(v))
    if not len(ok):
        return np.zeros(n, bool), np.zeros(n, bool)
    den = max(abs(v[ok[0]]), 0.5)
    dv = np.r_[0.0, np.diff(v)]
    dv = np.where(np.isfinite(dv), dv, 0.0)
    drop_seen = np.maximum.accumulate((dv / den <= -V_DROP) & reached)
    with np.errstate(invalid='ignore'):
        prev_low = np.r_[False, v[:-1] <= V_LOW * den]
    dt = np.r_[np.inf, np.diff(t)]
    rate = np.r_[0.0, np.diff(t1)] / dt
    rate = np.where(np.isfinite(rate), rate, 0.0)
    rapid_seen = np.maximum.accumulate((rate >= T1_RATE) & prev_low & reached)
    return drop_seen, rapid_seen


def sum_at_new_high(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """시간순으로 정렬된 한 실험. 각 행의 meas_t1 + meas_t2가 이전 행들의 최댓값 이상인지(6단계 제약). 값이 없는 행은 False."""
    s = t1 + t2
    finite = np.isfinite(s)
    prev_max = np.r_[-np.inf, np.maximum.accumulate(np.where(finite, s, -np.inf))[:-1]]
    with np.errstate(invalid='ignore'):
        return finite & (s >= prev_max)


def within_stage4_min(t: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """시간순으로 정렬된 한 실험. 예측이 처음 4단계 이상이 된 행부터 STAGE4_MIN_S초가 지나지 않은 행인지."""
    ge4 = np.flatnonzero(np.asarray(pred) >= 4)
    if not len(ge4):
        return np.zeros(len(t), bool)
    k = int(ge4[0])
    with np.errstate(invalid='ignore'):
        return (np.arange(len(t)) >= k) & (t - t[k] < STAGE4_MIN_S)


def apply_one(t, t1, v, pred, bat_cc: float, tr_method: int, use_rule1: bool = True,
              t2=None, use_peak_guard: bool = False, use_stage4_min: bool = False) -> np.ndarray:
    p = np.asarray(pred).copy()
    t = np.asarray(t, float)
    t1 = np.asarray(t1, float)
    drop_seen, rapid_seen = events(t, t1, np.asarray(v, float))
    if use_rule1:
        p[(p <= 3) & drop_seen] = 4
    if float(bat_cc) == 280.0 and int(tr_method) == 2:
        p[(p >= 5) & ~rapid_seen] = 4
        p[(p == 4) & rapid_seen] = 5
    if use_peak_guard:
        assert t2 is not None, '6단계 제약에는 meas_t2가 필요함'
        p[(p == 6) & sum_at_new_high(t1, np.asarray(t2, float))] = 5
    if use_stage4_min and float(bat_cc) <= SMALL_CC:
        p[(p >= 5) & within_stage4_min(t, p)] = 4
    return p


def apply(frame: pd.DataFrame, pred: np.ndarray, use_rule1: bool = True, use_peak_guard: bool = False,
          use_stage4_min: bool = False) -> np.ndarray:
    """frame의 행 순서로 주어진 예측을 고쳐 같은 순서로 돌려준다. 입력 행 순서와 무관하게 같은 결과가 나온다."""
    n = len(frame)
    out = np.asarray(pred).copy()
    if n == 0:
        return out
    num = lambda c: pd.to_numeric(frame[c], errors='coerce').to_numpy(float)   # noqa: E731
    t, t1, t2, v, cc, tm = num('relative_time_s'), num('meas_t1'), num('meas_t2'), num('meas_v'), num('bat_cc'), num('tr_method')
    exp = frame['experiment_id'].astype(str).to_numpy()
    order = np.lexsort((np.arange(n), t, exp))
    es = exp[order]
    starts = np.flatnonzero(np.r_[True, es[1:] != es[:-1]])
    ends = np.r_[starts[1:], n]
    for s, e in zip(starts, ends):
        idx = order[s:e]
        out[idx] = apply_one(t[idx], t1[idx], v[idx], out[idx], cc[idx[0]], tm[idx[0]], use_rule1, t2[idx], use_peak_guard, use_stage4_min)
    return out
