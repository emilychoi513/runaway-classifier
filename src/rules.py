"""Train·Validation에서 확인한 단계 기준(규칙 2개 + 6단계 제약 + 4단계 최소 지속 제약)을 모델 예측에 반영하는 후처리.

같은 experiment_id의 현재 행과 그 이전 행만 쓴다(미래 행·실험 전체 통계 미사용). 학습하는 값은 없고, 기준값은 아래 상수뿐이다.
"사건이 일어난 뒤 몇 초 지났나"가 아니라 "사건이 일어났나"만 본다. 경과 시간에 기대면 단계가 평소보다 길게 이어지는 실험에서 틀리기 때문이다.

규칙 1 (모든 실험): 전압 급락이 있었으면 4단계 이상
  - 전압 급락 = meas_t1이 50℃를 넘은 뒤, 한 행 사이에 meas_v가 처음 전압의 21% 이상 떨어짐.
  - 근거: 4단계가 있는 94개 실험 중 83개에서 이 사건과 4단계 시작이 1행 이내로 일치.
  - 모델이 3단계 이하로 본 행만 4단계로 올린다. 모델이 이미 4단계 이상으로 본 행은 건드리지 않는다.
  - 고치는 경우: 과충전 실험에서 전압이 무너졌다가 잠시 회복하면 모델이 3단계로 되돌아가는 문제.

규칙 2 (280Ah 과충전, bat_cc == 280 이고 tr_method == 2): 온도 급상승 전에는 5단계가 아니고, 급상승 뒤에는 4단계가 아님
  - 온도 급상승 = 전압이 이미 낮아진 상태(직전 행 meas_v가 처음 전압의 50% 이하)에서 meas_t1이 직전 행 대비 초당 1℃ 이상 오름.
  - 근거: 이 조건의 실험 7개 모두 5단계가 이 사건의 0~1행 전에 시작(Train 5개, Validation 2개).
  - 고치는 경우: 전압 붕괴 뒤 조용한 구간이 평소(75~165초)보다 길면 모델이 온도 수준·경과 시간만 보고 5단계를 일찍 선언하는 문제.
  - 28Ah도 같은 기준이 성립하지만(16개 모두 0~2행), 모델이 이미 더 정확해서 적용하지 않는다.

검증 (97개 실험 5-fold의 fold 밖 예측, 기반 모델 4가지): 매번 좋아진 실험 7~9개, 나빠진 실험 0개. 자세한 수치는 README.md.

6단계 제약 (모든 실험): 온도 합이 아직 최고치를 갱신하는 중이면 6단계가 아님
  - 온도 합 = meas_t1 + meas_t2. 이번 행의 온도 합이 같은 실험 이전 행들의 최댓값 이상이면 '갱신 중'.
  - 모델이 6단계로 본 행 중 '갱신 중'인 행만 5단계로 내린다. 다른 단계 예측은 건드리지 않는다.
  - 근거: 6단계 시작은 온도 합의 실험 전체 최고점과 93개 중 88개가 1행 이내로 일치. 아직 오르는 중인 행은 대부분 최고점 전(5단계)이다.
  - 고치는 경우: 온도 합이 급하게 오르는 도중에 C 모델이 6단계를 일찍 선언하는 문제.
  - 반대 방향("온도 합이 내려가면 6단계")은 5단계 중 잠깐 내려가는 실험이 많아 맞지 않는다(0.79). 그래서 내리기만 한다.
  - 검증: 97개 실험 5-fold, 기반 모델 4가지 모두 모의 점수 +0.0006, 바뀐 42행 중 28행이 맞게 바뀜, 좋아진 실험 13개·나빠진 실험 7개(1~2행씩).
    Train만 학습 → Validation(E_e5bcddc7 제외) 0.9320 → 0.9349. 이전 최댓값에 여유(0.5℃ 이상)를 두면 더 나빠진다.

4단계 최소 지속 제약 (소형 전지, bat_cc가 6Ah 이하): 4단계가 시작된 지 25초가 안 됐으면 5·6단계가 아님
  - 4단계 시작 = 같은 실험에서 (위 규칙·제약을 적용한) 예측이 처음 4단계 이상이 된 행. 그 행의 시각부터 25초가 지나지 않은 행의 5·6단계 예측만 4단계로 내린다.
  - 근거: 라벨이 있는 소형 전지(4.8Ah·6Ah) 실험 31개 모두 4단계가 30초 이상 이어진다(최솟값 30초인 실험 3개).
    4단계 시작 직후 전압이 0.5V 아래로 무너진 실험 6개에서도 4단계는 30~110초 이어졌다. 기준은 최솟값보다 한 행(5초) 짧은 25초로 잡았다.
  - 고치는 경우: 4단계 시작 직후 전압이 바로 무너지면 모델이 낮은 전압만 보고 30초가 되기 전에 5단계를 선언하는 문제.
  - 다른 전지에는 적용하지 않는다. 39·73Ah는 4단계가 0~20초로 짧고, 280Ah는 실험마다 달라 '최소 지속 시간'이 검증에서 유지되지 않았다.
    2·3·5단계의 최소 지속 시간도 같은 방식으로 검증했지만 틀리게 바꾸는 행이 더 많아 넣지 않았다.
  - 검증: 97개 실험 5-fold의 fold 밖 예측(시드 묶음 3개 × 5초 간격·솎은 입력)에서 바뀐 42행이 모두 맞게 바뀜. 매번 실험 2개가 좋아지고 나빠진 실험은 없음.
    모의 점수 +0.0007~+0.0008. Train만 학습 → Validation에서는 바뀌는 행이 없다(점수 그대로).
  - 이 제약을 찾은 경위는 README.md의 "학습·검증 구성"에 적었다.
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
    """시간순으로 정렬된 한 실험. 예측이 처음 4단계 이상이 된 행의 시각부터 STAGE4_MIN_S초가 지나지 않은 행인지(4단계 최소 지속 제약).

    처음 4단계 이상이 된 행은 그 행까지의 예측만으로 정해지므로, 각 행의 값은 이후 행과 무관하다.
    """
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
