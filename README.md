# Uni-D 배터리 열폭주 6단계 분류 — 2팀 제출 코드

- 팀명: 2팀
- 최종 선택 제출 파일: `submission.csv` (Public 0.9723). `test.py`로 그대로 다시 만들 수 있습니다.

## 코드 실행 방법
```bash
pip install -r requirements.txt

# 추론: 저장된 가중치(weights/)로 제출 파일을 다시 만듭니다
python test.py --input <battery_data_light>/test/test.csv --image_root <battery_data_light>/test --output submission.csv

# 학습: 제출 가중치를 다시 만듭니다 (Train+Validation → weights/, CPU로 2분 안팎)
python train.py --data_dir <battery_data_light>

# 검증: Train만 학습하고 Validation 점수를 출력합니다 (weights_train/ 에 저장)
python train.py --data_dir <battery_data_light> --train_only

# 제출 전 점검 (대회 제공 스크립트)
python check_submission.py --data_dir <battery_data_light> --final submission.csv
```
- `--image_root`에는 `images/` 폴더의 상위 폴더(`test`)를 줍니다. 이미지 폴더 자체(`test/images`)를 줘도 결과가 같습니다.
- 출력 행 순서는 입력 CSV 순서입니다. `--sample_submission <battery_data_light>/sample_submission.csv`를 주면 제출 양식 순서로 저장합니다.
- 시드를 고정했고 CPU로만 학습하므로, 다시 학습한 가중치로도 같은 예측이 나옵니다. 모델 파일의 바이트는 저장할 때마다 달라집니다.

| 파일 | 역할 |
|---|---|
| `model.py` | 피처 표 구성, 모델 설정, 확률 결합, 추론 함수 |
| `train.py` | 학습하고 가중치를 `weights/`에 저장 |
| `test.py` | 가중치를 불러와 예측하고 CSV 저장 |
| `weights/` | CatBoost 모델 9개와 설정(`meta.json`) |
| `src/sensor_features.py` | 표 피처: 센서·조건·가스 값과 과거 방향 피처 |
| `src/image_features.py` | 열화상 이미지 피처 |
| `src/pressure_features.py` | 압력 변곡점 피처 (모델 B) |
| `src/temp_sum_features.py` | 온도 합 최고점 피처 (모델 C) |
| `src/rules.py` | 예측 뒤에 적용하는 규칙 |
| `check_submission.py` | 대회 제공 점검 스크립트 (수정 없음) |
| `submission.csv` | 최종 선택 제출 파일 |

## 개발 환경
- OS: Ubuntu 22.04.5 LTS
- Python: 3.10.12
- GPU: 대회 제공 서버의 NVIDIA GeForce RTX 5080. 학습과 추론은 CPU만 씁니다.
- 라이브러리: catboost 1.2.10, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.15.3, pillow 12.3.0 (`requirements.txt`)

## 모델 구성
| 구성 | 내용 |
|---|---|
| A | 6단계 CatBoost (depth 8, 시드 3개 평균). 입력: 표 피처 41개 + 이미지 피처 4개 |
| B | 2↔3단계 전용 CatBoost (2·3단계 행만으로 학습). 입력: 표 피처 41개 + 압력 변곡점 피처 48개 |
| C | 5↔6단계 전용 CatBoost (5·6단계 행만으로 학습). 입력: 표 피처 41개 + 온도 합 피처 15개 |
| 결합 | A의 (2+3단계) 확률 합을 B의 비율로, (5+6단계) 확률 합을 C의 비율로 나눈 뒤 확률이 가장 높은 단계를 고름 |
| 규칙 | 결합한 예측에 아래 네 가지를 차례로 적용 (`src/rules.py`) |

규칙은 모두 현재 행과 이전 행만 보고, 학습하는 값이 없습니다.
1. **전압 급락이 있었으면 4단계 이상** (모든 실험). 급락은 `meas_t1`이 50℃를 넘은 뒤 한 행 사이에 `meas_v`가 처음 전압의 21% 이상 떨어지는 것입니다.
2. **온도 급상승 전에는 5단계가 아니고, 급상승 뒤에는 4단계가 아님** (280Ah 과충전). 급상승은 전압이 이미 낮아진 상태에서 `meas_t1`이 초당 1℃ 이상 오르는 것입니다.
3. **6단계 제약** (모든 실험). `meas_t1 + meas_t2`가 아직 최고치를 갱신하는 중이면 6단계 예측을 5단계로 내립니다.
4. **4단계 최소 지속 제약** (6Ah 이하 전지). 4단계가 시작된 지 25초가 안 된 행은 5·6단계 예측을 4단계로 내립니다.

## 학습·검증 데이터 구성
- 학습: Train + Validation으로 제출 가중치를 학습했습니다. Validation 실험 1개(`E_e5bcddc7`)는 4단계 없이 5단계가 길게 이어져 다른 실험들의 기준과 어긋나므로 넣지 않았습니다.
- 간격 증강: 모든 실험을 평균 12초 간격(10·15초)으로 솎은 복사본을 원본(5초 간격)과 함께 학습했습니다. Test 일부 실험의 관측 간격이 넓기 때문입니다.
- 검증: (1) Train만 학습한 같은 구성의 모델로 제공 Validation을 채점, (2) Train+Validation 97개 실험의 실험 단위 5-fold. 두 방법 모두 채점하는 실험을 학습에 쓰지 않습니다.
- Test는 학습에 쓰지 않았습니다.

| 검증 점수 (Macro F1) | A만 | A + B + C | + 규칙 1·2 | + 6단계 제약 | + 4단계 최소 지속 제약 (최종) |
|---|---|---|---|---|---|
| Train만 학습 → Validation 전체 | 0.7645 | 0.7819 | 0.7830 | 0.7857 | 0.7857 |
| Train만 학습 → Validation, `E_e5bcddc7` 제외 | 0.9110 | 0.9305 | 0.9320 | 0.9349 | 0.9349 |
| 97개 실험 5-fold, Test와 같은 조건 구성으로 채점 | — | 0.9288 | 0.9338 | 0.9343 | 0.9350 |

- Validation 두 행은 `python train.py --train_only`가 출력하는 값입니다. 5-fold 행은 이 레포의 피처·모델 코드를 불러 쓰는 별도 분석 스크립트로 계산했고, 그 스크립트는 레포에 넣지 않았습니다.
- Validation 전체 점수가 낮은 것은 학습에서 뺀 실험 `E_e5bcddc7`(1,543행)에서 오류가 대부분 나오기 때문입니다.
- 규칙의 기준값은 Train+Validation 분석으로 정했으므로, 위 Validation 점수는 기준값에 대해서는 완전히 독립된 점수가 아닙니다.
- 규칙은 모두 위 검증에서 점수가 내려가지 않는 것을 확인한 뒤 리더보드에 제출했고, Public 점수가 오른 것을 확인한 뒤 최종 구성에 넣었습니다.
- 4단계 최소 지속 제약을 찾은 경위: Test에 대한 모델의 예측에서 4.8Ah 실험의 4단계가 유난히 짧게 나온 것을 보고 점검을 시작했습니다. Public과 Private 양쪽 점수에 영향이 있는 변경을 찾던 중이었고, 그때 리더보드 점수 변화로 어떤 실험이 Public에 들어 있는지 추정해 참고했습니다. 기준(6Ah 이하 전지, 25초)은 Train·Validation 라벨 통계(해당 전지 실험 31개 모두 4단계가 30초 이상)로 정했습니다. Test 정답은 보지 않았습니다.

## 시계열 피처 구성
모두 같은 `experiment_id` 안에서 `relative_time_s`가 현재 이하인 관측만 씁니다. 창 길이는 행 수가 아니라 초 단위입니다.
- 온도(`meas_t1`, `meas_t2`, `thermal_t`): 지금까지의 최댓값, 최댓값에서 내려온 정도, 10·30·60·120초 전 대비 변화, 최댓값을 마지막으로 갱신한 뒤 지난 초
- 전압(`meas_v`): 처음 값 대비 변화, 지금까지의 최솟값, 5초 전 대비 변화, 처음 대비 0.77V 이상 떨어진 뒤 지난 초
- 사건 여부와 경과 시간: `meas_t1`이 45℃에 닿았는지와 그 뒤 지난 초
- 가스 10종 합: 지금까지의 최댓값, 30초 전 대비 변화
- 압력(`meas_p`): 과거 120초·300초 구간에 맞춘 추세에서 벗어난 정도, 15·30·60·120초 변화, 그 누적 최댓값, 45℃·50℃ 도달 시점 대비 변화
- 온도 합(`meas_t1 + meas_t2`): 지금까지의 최댓값, 최댓값에서 내려온 정도, 최댓값 갱신 뒤 지난 초, 5~60초 변화
- 이미지: 열화상 평균 수준과 뜨거운 픽셀 비율의 지금까지 최댓값

미래 시점 관측과 실험 전체 기준 통계(전체 평균·최댓값·길이·순위)는 쓰지 않습니다. `relative_time_s`, `indu_t`, `indu_t.1`, `meas_t1_plus_t2`는 모델 입력에 넣지 않았습니다.
입력 데이터로 추정하는 전처리(표준화, 결측 대체)는 없습니다. 결측은 그대로 CatBoost에 넘깁니다.

## 사용한 사전학습 모델
없습니다. 모든 모델(CatBoost)은 대회 제공 데이터로 처음부터 학습했습니다. 따라서 적을 출처·버전·라이선스가 없습니다.

## 재현 확인
- `check_submission.py`의 점검 7개를 모두 통과했습니다. 모든 이미지를 검은 이미지로 바꾸면 3,451행 중 0.55%의 예측이 바뀝니다.
- 새로 받은 레포에서 `train.py`로 다시 학습한 뒤 `test.py`를 실행해도 `submission.csv`와 바이트 단위로 같은 파일이 나옵니다.
- `weights/meta.json`의 `use_rule1`, `use_peak_guard`, `use_stage4_min`으로 규칙 1, 6단계 제약, 4단계 최소 지속 제약을 끌 수 있습니다. 제출 파일을 재현하려면 셋 다 `true`로 둡니다.
