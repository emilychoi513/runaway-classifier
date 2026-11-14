# Uni-D 배터리 열폭주 6단계 분류 — 2팀 제출 코드

## 팀 정보
- 팀명: 2팀
- 최종 선택 제출 파일: `submission_stage8_peak_guard_trainval.csv` (Public 0.9645), Private 복원 대상
- 이전 제출 `submission_stage6_rules_trainval.csv`(Public 0.9555)는 6단계 제약만 뺀 버전입니다. `weights/meta.json`의 `use_peak_guard`를 `false`로 바꾸면 재현됩니다.

## 개발 환경
- OS: Ubuntu 22.04.5 LTS
- Python: 3.10.12
- GPU: 대회 제공 GPU 서버 (NVIDIA GeForce RTX 5080). 학습과 추론은 CPU만 씁니다.
- 라이브러리: `requirements.txt` 참고

## 파일 구성
| 파일 | 역할 |
|---|---|
| `model.py` | 피처 표 구성, 모델 설정, 확률 결합, 추론 함수. `train.py`와 `test.py`가 함께 씁니다 |
| `train.py` | Train으로 학습하고 Validation으로 평가, 가중치 저장. `--refit_train_val`을 주면 Train+Validation으로 학습 |
| `test.py` | 입력 CSV에 들어 있는 행만으로 추론하고 제출 파일 생성 |
| `weights/` | 학습된 가중치(CatBoost 모델 9개)와 `meta.json`. 재학습 없이 `test.py`만으로 결과를 재현합니다 |
| `src/fg_features.py`, `src/features.py` | 표 피처: 센서·조건·가스 값과 과거 방향 피처 |
| `src/image_features.py` | 열화상 이미지 피처 (색을 온도 수준으로 바꾼 통계) |
| `src/pfeat.py` | 압력 변곡점 피처 48개 (2↔3단계 구분용) |
| `src/sfeat.py` | 온도 합(`meas_t1 + meas_t2`) 최고점 피처 15개 (5↔6단계 구분용) |
| `src/rules.py` | 예측 뒤에 적용하는 단계 기준 규칙 2개와 6단계 제약 |
| `check_submission.py` | 대회 템플릿의 제출 전 점검 스크립트 (수정 없음) |
| `submission_stage8_peak_guard_trainval.csv` | 최종 선택 제출 파일 |
## 실행 방법
```bash
pip install -r requirements.txt

# 추론 (제출 파일 재현). 가중치는 weights/ 를 씁니다
python test.py --input <battery_data_light 경로>/test/test.csv \
               --image_root <battery_data_light 경로>/test \
               --output submission.csv

# 학습: Train만 학습하고 Validation 점수 확인
python train.py --data_dir <battery_data_light 경로> --model_out weights_train

# 학습: Train+Validation으로 최종 학습 (weights/ 를 다시 만들 때)
python train.py --data_dir <battery_data_light 경로> --model_out weights --refit_train_val

# 제출 전 점검
python check_submission.py --data_dir <battery_data_light 경로> --final submission_stage8_peak_guard_trainval.csv
```
- `--image_root`에는 `test` 폴더(CSV의 `image_path`가 가리키는 `images/` 폴더의 상위 폴더)를 줍니다. `test/images`처럼 이미지 폴더 자체를 줘도 같은 이미지를 찾아 같은 결과가 나옵니다. 이미지를 하나도 찾지 못하면 오류를 냅니다.
- 출력 행 순서는 입력 CSV 순서입니다. `--sample_submission <battery_data_light 경로>/sample_submission.csv`를 주면 그 양식의 순서로 저장합니다(예측은 같음).
- `train.py`에 `--cache_dir <폴더>`를 주면 이미지 피처를 저장해 두고 다시 씁니다.

## 모델 구조
| 구성 | 내용 |
|---|---|
| A | 6단계 CatBoost (MultiClass, depth 8, 시드 3개 평균). 입력: 표 피처 41개 + 이미지 피처 4개 |
| B | 2↔3단계 전용 CatBoost (2·3단계 행만으로 학습, depth 6, 트리 300, 시드 3개). 입력: 표 피처 41개 + 압력 변곡점 피처 48개 |
| C | 5↔6단계 전용 CatBoost (5·6단계 행만으로 학습, depth 6, 트리 300, 시드 3개). 입력: 표 피처 41개 + 온도 합 피처 15개 |
| 결합 | 1·4단계 확률은 A 그대로. A의 (2+3단계) 확률 합을 B의 비율로, (5+6단계) 확률 합을 C의 비율로 나눈 뒤 확률이 가장 높은 단계를 고름 |
| 규칙 | 결합한 예측에 아래 두 규칙과 6단계 제약을 적용 |

규칙 (`src/rules.py`). 모두 현재 행과 이전 행만 보고, 학습하는 값이 없습니다.
1. **전압 급락이 있었으면 4단계 이상** (모든 실험). 급락은 `meas_t1`이 50℃를 넘은 뒤 한 행 사이에 `meas_v`가 처음 전압의 21% 이상 떨어지는 것입니다. 모델이 3단계 이하로 본 행만 올립니다.
2. **온도 급상승 전에는 5단계가 아니고, 급상승 뒤에는 4단계가 아님** (280Ah 과충전: `bat_cc` 280, `tr_method` 2). 급상승은 전압이 이미 낮아진 상태에서 `meas_t1`이 초당 1℃ 이상 오르는 것입니다.
3. **6단계 제약: 온도 합이 아직 최고치를 갱신하는 중이면 6단계가 아님** (모든 실험). 이번 행의 `meas_t1 + meas_t2`가 같은 실험 이전 행들의 최댓값 이상이면, 모델이 6단계로 본 행을 5단계로 내립니다. 다른 단계는 건드리지 않습니다. 6단계 시작은 온도 합의 최고점과 93개 실험 중 88개가 1행 이내로 일치하므로, 아직 오르는 중인 행은 대부분 최고점 전(5단계)입니다.

## 학습·검증 구성
- 학습 데이터: Train + Validation. Validation 실험 `E_e5bcddc7` 하나는 넣지 않았습니다. 전압이 0이 되자마자 4단계 없이 5단계가 2,950초 이어지는 실험으로, 용량이 같은 다른 실험들의 4·5단계 기준과 어긋납니다.
- 간격 증강: 원본(5초 간격)에 더해, 모든 실험을 평균 12초 간격(10·15초)으로 솎은 복사본을 함께 학습했습니다. Test 일부 실험의 관측 간격이 넓기 때문입니다.
- 검증 방법: (1) 제공 Validation — Train만 학습한 같은 구성의 모델로 채점, (2) Train+Validation 97개 실험의 실험 단위 5-fold.
- 변경을 채택한 기준: 5-fold와 Validation에서 모두 오르고, 실험 단위로 나빠진 실험보다 좋아진 실험이 많을 것. Public 점수로 고르지 않았습니다.

| 검증 점수 (Macro F1) | A만 | A + B + C | + 규칙 1·2 | + 6단계 제약 |
|---|---|---|---|---|
| Train만 학습 → Validation 전체 | 0.7645 | 0.7819 | 0.7830 | 0.7857 |
| Train만 학습 → Validation, `E_e5bcddc7` 제외 | 0.9110 | 0.9305 | 0.9320 | 0.9349 |
| 97개 실험 5-fold, Test와 같은 조건 구성으로 채점 | 0.915 | 0.933 | 0.9363 | 0.9369 |

6단계 제약 검증: 5-fold에서 기반 모델 4가지 모두 +0.0006, 바뀐 42행 중 28행이 맞게 바뀜(모두 6 → 5단계), 좋아진 실험 13개·나빠진 실험 7개(각 1~2행),
fold 5개 모두 0 이상. Validation에서는 바뀐 8행 중 7행이 맞음. 이전 최댓값에 0.5℃ 이상 여유를 두면 더 나빠져서 여유 없이 씁니다.

Validation 전체 점수가 낮은 것은 `E_e5bcddc7`(1,543행) 하나에서 오류가 대부분 나오기 때문입니다.

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
- 없음

## 규정 준수·재현 확인 (이 폴더, 2026-11-15)
| 확인 | 결과 |
|---|---|
| `check_submission.py --final submission_stage8_peak_guard_trainval.csv`: 형식, 같은 입력 재실행, 최종 파일 재현, 행 순서 섞기, 실험 나눠 넣기, 미래 행 빼고 넣기 | 모두 통과 (다른 예측 0행) |
| `check_submission.py`: 모든 이미지를 검은 이미지로 교체 | 예측이 바뀜 (3,451행 중 0.55%) |
| `--image_root`에 `test` / `test/images` / `battery_data_light`를 줌 | 세 경우 모두 최종 선택 파일과 바이트 단위로 동일 |
| `--image_root`에 이미지가 없는 폴더를 줌 | 예측하지 않고 오류를 냄 (이미지 피처가 빈 값인 채로 조용히 예측하지 않음) |
| `weights/meta.json`의 `use_peak_guard`를 `false`로 바꾸고 `test.py` 실행 | `submission_stage6_rules_trainval.csv`(0.9555)와 바이트 단위로 동일 |
| 0.9555 제출 파일과 비교 | 3행만 다름, 모두 6 → 5단계: `E_8c92cdf4` 1270·1275초, `E_c8e73087` 5680초 |

가중치는 0.9555 제출 때와 같은 파일입니다(sha256 동일). 그때 `train.py --refit_train_val`로 다시 학습한 가중치도 규칙 1·2 적용 후 3,451행 모두 같은 예측을 냈고, 6단계 제약은 그 예측에만 적용되므로 재학습 재현은 이 버전에서도 그대로 성립합니다.

## 참고: 규칙 켜고 끄기
- `weights/meta.json`의 `"use_peak_guard": false` → 6단계 제약을 끕니다. 출력이 0.9555 제출 파일과 같아집니다.
- `"use_rule1": false` → 규칙 1을 끕니다. 실험 `E_753e27ba`의 1115~1160초 10행이 4단계 → 3단계로 바뀝니다.
- 이 폴더의 제출 파일을 재현하려면 둘 다 `true`로 두어야 합니다.
