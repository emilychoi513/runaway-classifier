"""Uni-D 배터리 열폭주 6단계 분류 — 2팀 공통 모듈 (피처 생성, 모델 구성, 결합, 규칙).

train.py와 test.py가 모두 이 모듈을 쓴다. 피처 계산식은 src/ 아래에 있다.

구성
- A: 6단계 CatBoost. 표 피처 41개 + 열화상 이미지 피처 4개.
- B: 2↔3단계 전용 CatBoost. 표 피처 41개 + 압력 변곡점 피처 48개 (src/pfeat.py).
- C: 5↔6단계 전용 CatBoost. 표 피처 41개 + 온도 합(meas_t1 + meas_t2) 최고점 피처 15개 (src/sfeat.py).
- 결합: 1·4단계 확률은 A 그대로. A의 (2단계+3단계) 확률 합을 B의 비율로, (5단계+6단계) 확률 합을 C의 비율로 나눈다.
- 규칙: 결합한 예측 뒤에 단계 기준 규칙 2개, 6단계 제약, 4단계 최소 지속 제약(소형 전지)을 적용한다 (src/rules.py).
  규칙 1, 6단계 제약, 4단계 최소 지속 제약은 weights/meta.json의 use_rule1·use_peak_guard·use_stage4_min으로 켜고 끈다.

규정 준수
- 모든 과거 방향 피처와 규칙은 같은 experiment_id 안에서 현재 행과 그 이전 행만 쓴다. 창 길이는 초 단위(relative_time_s 기준)다.
- 실험 전체 통계(전체 평균·최댓값·길이·순위)는 쓰지 않는다. 입력 데이터로 추정하는 전처리(표준화, 결측 대체)도 없다.
- 입력 금지 컬럼: sample_id, experiment_id, relative_time_s, image_path, label, indu_t, indu_t.1, meas_t1_plus_t2
  (relative_time_s는 과거 방향 창을 정하는 데만 쓰고 모델 입력으로는 넣지 않는다).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

import features as FT  # noqa: E402  표 피처 (센서·조건·가스·과거 방향)
import image_features as IM  # noqa: E402  열화상 이미지 피처
import pfeat  # noqa: E402  압력 변곡점 피처
import rules  # noqa: E402  단계 기준 규칙
import sfeat  # noqa: E402  온도 합 최고점 피처

LABELS = [1, 2, 3, 4, 5, 6]
SEEDS = [42, 1, 2]

# 표 피처 41개: 온도 2, 배터리 조건 4, 가스 10 + 가스 전체 0 여부, 과거 방향 피처 24
BASE41 = [
    'thermal_t', 'atmp_t', 'bat_class', 'bat_cc', 'soc_status', 'tr_method',
    'meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf', 'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2', 'gas_all_zero',
    'past_t1_cummax', 'past_t1_drop_from_max', 'past_v_change_from_start', 'past_v_cummin', 'past_t1_change_30s', 'past_v_change_5s',
    'past_s_since_t1_45', 'past_s_since_v_drop', 'past_t1_reached_45', 'past_v_dropped',
    'past_t1_rise_from_start', 'past_t1_change_10s', 'past_t1_change_60s', 'past_t1_change_120s', 'past_t1_max_change_30s', 'past_s_since_t1_max',
    'past_t2_cummax', 'past_t2_drop_from_max', 'past_t2_change_30s',
    'past_thermal_cummax', 'past_thermal_drop_from_max', 'past_thermal_change_30s',
    'past_gas_sum_cummax', 'past_gas_sum_change_30s',
]
A_COLS = BASE41 + IM.IC_COLS
B_COLS = BASE41 + pfeat.P_COLS
C_COLS = BASE41 + sfeat.S_COLS

A_PARAMS = {'loss_function': 'MultiClass', 'learning_rate': 0.05, 'depth': 8, 'task_type': 'CPU', 'nan_mode': 'Min'}
A_TREES = {42: 247, 1: 273, 2: 260}      # 시드별 트리 수. Train 5-fold의 fold 밖 loss가 가장 낮았던 값
B_PARAMS = {'loss_function': 'Logloss', 'learning_rate': 0.05, 'depth': 6, 'task_type': 'CPU', 'nan_mode': 'Min'}
B_TREES = 300
C_PARAMS = dict(B_PARAMS)
C_TREES = 300

# Train+Validation으로 최종 학습할 때 넣지 않는 Validation 실험.
# 전압이 0이 되자마자 4단계 없이 5단계가 2,950초 이어지는 실험으로, 용량이 같은 다른 실험들의 4·5단계 기준과 어긋난다.
# 이 실험을 학습에 넣으면 Train+Validation 5-fold에서 다른 실험의 '4단계를 5단계로' 오류가 70 → 97행으로 늘었다.
EXCLUDE = ['E_e5bcddc7d77f459ea541cf658ab4346e']

# 간격 증강: 원본(5초 간격)에 더해 평균 12초 간격(10·15초)으로 솎은 복사본을 함께 학습한다. Test 일부 실험의 간격이 넓기 때문이다.
THIN_AVG_GAP_S, THIN_STEP_S = 12.0, 5.0


# ---------------------------------------------------------------------------
# 피처 표
# ---------------------------------------------------------------------------
def build_table(frame: pd.DataFrame, image_root, cache_file=None) -> pd.DataFrame:
    """frame에 들어 있는 행만으로 모든 피처를 만든다. 행 순서는 입력과 같다.

    cache_file은 학습 때 이미지 현재 행 피처를 저장·재사용하는 용도다(test.py는 쓰지 않는다).
    """
    table = pd.concat([IM.full_table(frame, image_root, cache_file=cache_file), pfeat.p_features(frame), sfeat.s_features(frame)], axis=1)
    assert table.columns.is_unique
    return table


def _keep_index(n: int) -> np.ndarray:
    """[학습 전용] n행짜리 실험에서 남길 행 번호. 등간격으로 골라 평균 간격이 THIN_AVG_GAP_S초가 되게 한다."""
    m = int(np.floor((n - 1) * THIN_STEP_S / THIN_AVG_GAP_S)) + 1
    return np.floor(np.linspace(0, n - 1, m)).astype(int)


def thin_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """[학습 전용] 실험마다 행을 솎아 10·15초 간격으로 만든 복사본."""
    parts = [g.iloc[_keep_index(len(g))]
             for _, g in frame.sort_values(['experiment_id', 'relative_time_s'], kind='mergesort').groupby('experiment_id', sort=False)]
    return pd.concat(parts).reset_index(drop=True)


def thin_table(thin: pd.DataFrame, orig: pd.DataFrame, orig_table: pd.DataFrame) -> pd.DataFrame:
    """[학습 전용] 솎은 행만으로 과거 방향 피처를 다시 계산한 피처 표. 이미지 현재 행 피처는 같은 이미지라 원본 값을 쓴다."""
    ir = orig_table[IM.IR_COLS].set_index(orig['sample_id'].to_numpy()).loc[thin['sample_id'].to_numpy()].reset_index(drop=True)
    table = pd.concat([FT.feature_table(thin), ir, IM.past_table(thin, ir), pfeat.p_features(thin), sfeat.s_features(thin)], axis=1)
    assert list(table.columns) == list(orig_table.columns)
    return table


def to32(table: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """피처 표에서 cols를 골라 모델 입력 행렬(float32)로 만든다."""
    return table[cols].to_numpy(dtype=np.float64).astype(np.float32)


# ---------------------------------------------------------------------------
# 결합과 추론
# ---------------------------------------------------------------------------
def compose(prob_a: np.ndarray, r3: np.ndarray, r6: np.ndarray) -> np.ndarray:
    """A의 2·3단계 확률 합을 B의 3단계 비율 r3로, 5·6단계 확률 합을 C의 6단계 비율 r6로 다시 나눈다. 1·4단계 확률은 A 그대로."""
    out = prob_a.copy()
    s = prob_a[:, 1] + prob_a[:, 2]
    out[:, 1] = s * (1.0 - r3)
    out[:, 2] = s * r3
    s = prob_a[:, 4] + prob_a[:, 5]
    out[:, 4] = s * (1.0 - r6)
    out[:, 5] = s * r6
    return out


def load_meta(model_dir) -> dict:
    """가중치 폴더의 meta.json(피처 구성, 규칙 스위치, 모델 파일 목록)."""
    return json.loads((Path(model_dir) / 'meta.json').read_text())


def predict_from_table(frame: pd.DataFrame, table: pd.DataFrame, model_dir, threads: int = 4) -> np.ndarray:
    """저장된 모델로 단계(1~6)를 예측한다. 시드 평균 → 결합 → 가장 높은 단계 → 규칙."""
    model_dir = Path(model_dir)
    meta = load_meta(model_dir)
    assert meta['A_columns'] == A_COLS and meta['B_columns'] == B_COLS and meta['C_columns'] == C_COLS, '피처 구성이 학습 때와 다름'
    assert meta['image_feature_version'] == IM.VERSION and meta['rules_version'] == rules.VERSION, '피처·규칙 버전이 학습 때와 다름'
    xa, xb, xc = to32(table, A_COLS), to32(table, B_COLS), to32(table, C_COLS)
    prob_a, r3, r6 = np.zeros((len(frame), len(LABELS))), np.zeros(len(frame)), np.zeros(len(frame))
    for m in meta['models']:
        a, b, c = CatBoostClassifier(), CatBoostClassifier(), CatBoostClassifier()
        a.load_model(str(model_dir / m['A']['file']))
        b.load_model(str(model_dir / m['B']['file']))
        c.load_model(str(model_dir / m['C']['file']))
        assert list(a.feature_names_) == A_COLS and [int(k) for k in a.classes_] == LABELS and a.tree_count_ == m['A']['trees']
        assert list(b.feature_names_) == B_COLS and [int(k) for k in b.classes_] == [0, 1] and b.tree_count_ == m['B']['trees']
        assert list(c.feature_names_) == C_COLS and [int(k) for k in c.classes_] == [0, 1] and c.tree_count_ == m['C']['trees']
        prob_a += a.predict_proba(xa, thread_count=threads) / len(meta['models'])
        r3 += b.predict_proba(xb, thread_count=threads)[:, 1] / len(meta['models'])
        r6 += c.predict_proba(xc, thread_count=threads)[:, 1] / len(meta['models'])
    pred = np.asarray(LABELS)[compose(prob_a, r3, r6).argmax(1)]
    return rules.apply(frame, pred, use_rule1=bool(meta['use_rule1']), use_peak_guard=bool(meta['use_peak_guard']),
                       use_stage4_min=bool(meta.get('use_stage4_min', False)))


def predict(frame: pd.DataFrame, image_root, model_dir, threads: int = 4) -> np.ndarray:
    """입력 CSV의 행만으로 피처를 만들고 예측한다. 이미지는 image_root 아래에서 매번 새로 읽는다(캐시 없음)."""
    assert frame['sample_id'].is_unique, 'sample_id 중복'
    return predict_from_table(frame, build_table(frame, image_root, cache_file=None), model_dir, threads)


def macro_f1(y, pred) -> float:
    """6개 단계 F1의 평균(대회 지표)."""
    return float(f1_score(y, pred, labels=LABELS, average='macro', zero_division=0))
