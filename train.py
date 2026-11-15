#!/usr/bin/env python
"""학습 스크립트: 모델 A·B·C를 학습해 가중치 폴더에 저장한다. Test는 읽지 않는다.

예) python train.py --data_dir <battery_data_light>                # 제출 가중치 재현: Train+Validation 학습 → weights/
    python train.py --data_dir <battery_data_light> --train_only   # 검증용: Train만 학습 → weights_train/, Validation 점수 출력

- 기본 실행은 제출에 쓴 구성 그대로다. 시드를 고정했고 CPU로만 학습하므로 다시 학습해도 같은 예측이 나온다.
- 세 모델 모두 원본 행과 평균 12초 간격으로 솎은 복사본으로 학습한다.
- --no_rule1 / --no_peak_guard / --no_stage4_min 은 해당 규칙을 끈 설정으로 저장한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import model as M


def load_split(data_dir: Path, split: str, cache_dir):
    frame = pd.read_csv(data_dir / split / f'{split}.csv')
    cache = None if cache_dir is None else Path(cache_dir) / f'image_row_{split}.npz'
    return frame, M.build_table(frame, data_dir / split, cache_file=cache)


def with_thinned(frame: pd.DataFrame, table: pd.DataFrame):
    """원본 + 솎은 복사본의 (A 입력, B 입력, C 입력, 라벨, 복사본 행 수)."""
    thin = M.thin_frame(frame)
    assert set(thin.groupby('experiment_id')['relative_time_s'].diff().dropna().unique()) <= {10.0, 15.0}
    t_thin = M.thin_table(thin, frame, table)
    y = np.concatenate([frame['label'].to_numpy(), thin['label'].to_numpy()])
    x = [np.concatenate([M.to32(table, cols), M.to32(t_thin, cols)]) for cols in (M.A_COLS, M.B_COLS, M.C_COLS)]
    return x[0], x[1], x[2], y, len(thin)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_dir', type=Path, default=Path('data/battery_data_light'),
                    help='train/, validation/ 폴더가 있는 battery_data_light 경로 (기본: data/battery_data_light)')
    ap.add_argument('--model_out', type=Path, default=None, help='가중치를 저장할 폴더 (기본: weights, --train_only면 weights_train)')
    ap.add_argument('--train_only', action='store_true',
                    help='Train만 학습하고 Validation(학습에 쓰지 않음)으로 채점. 기본 저장 폴더가 weights_train이라 제출 가중치를 덮어쓰지 않음')
    ap.add_argument('--refit_train_val', action='store_true', help='Train+Validation으로 학습 (기본 동작과 같음. 이전 명령과의 호환용)')
    ap.add_argument('--no_rule1', action='store_true', help='규칙 1(전압 급락 → 4단계 이상)을 끈 설정으로 저장')
    ap.add_argument('--no_peak_guard', action='store_true', help='6단계 제약(온도 합 최고치 갱신 중이면 6단계 아님)을 끈 설정으로 저장')
    ap.add_argument('--no_stage4_min', action='store_true', help='4단계 최소 지속 제약(소형 전지: 4단계 시작 뒤 25초 안에는 5단계 아님)을 끈 설정으로 저장')
    ap.add_argument('--cache_dir', type=Path, default=None, help='이미지 현재 행 피처를 저장·재사용할 폴더 (없으면 매번 이미지에서 계산)')
    ap.add_argument('--threads', type=int, default=6)
    args = ap.parse_args()
    if args.train_only and args.refit_train_val:
        ap.error('--train_only 와 --refit_train_val 은 함께 쓸 수 없습니다.')
    if not (args.data_dir / 'train' / 'train.csv').is_file():
        ap.error(f'{args.data_dir}/train/train.csv 가 없습니다. --data_dir 에 battery_data_light 경로를 주세요.')
    refit = not args.train_only   # 기본: 제출 구성(Train+Validation)
    out = args.model_out if args.model_out is not None else M.ROOT / ('weights' if refit else 'weights_train')
    out.mkdir(parents=True, exist_ok=True)
    use_rule1, use_peak_guard, use_stage4_min = not args.no_rule1, not args.no_peak_guard, not args.no_stage4_min

    train, t_train = load_split(args.data_dir, 'train', args.cache_dir)
    val, t_val = load_split(args.data_dir, 'validation', args.cache_dir)
    assert not (set(train['experiment_id']) & set(val['experiment_id']))
    parts = [(train, t_train)]
    if refit:
        keep = ~val['experiment_id'].isin(M.EXCLUDE).to_numpy()
        assert (~keep).sum() > 0
        parts.append((val[keep].reset_index(drop=True), t_val[keep].reset_index(drop=True)))
    xa, xb, xc, y, n_thin = [], [], [], [], 0
    for frame, table in parts:
        a, b, c, yy, nt = with_thinned(frame, table)
        xa.append(a); xb.append(b); xc.append(c); y.append(yy); n_thin += nt
    xa, xb, xc, y = np.concatenate(xa), np.concatenate(xb), np.concatenate(xc), np.concatenate(y)
    n_orig = sum(len(f) for f, _ in parts)
    m23, m56 = np.isin(y, [2, 3]), np.isin(y, [5, 6])
    print(f'학습 행: 원본 {n_orig} + 솎은 복사본 {n_thin} | B 학습 행(2·3단계) {int(m23.sum())} | C 학습 행(5·6단계) {int(m56.sum())}', flush=True)

    xa_val, xb_val, xc_val = M.to32(t_val, M.A_COLS), M.to32(t_val, M.B_COLS), M.to32(t_val, M.C_COLS)
    y_val = val['label'].to_numpy()
    models, pa_val, r3_val, r6_val = [], [], [], []
    t0 = time.time()
    common = dict(thread_count=args.threads, allow_writing_files=False, verbose=0)
    for seed in M.SEEDS:
        a = CatBoostClassifier(**M.A_PARAMS, random_seed=seed, iterations=M.A_TREES[seed], **common)
        a.fit(Pool(xa, y, feature_names=M.A_COLS))
        b = CatBoostClassifier(**M.B_PARAMS, random_seed=seed, iterations=M.B_TREES, **common)
        b.fit(Pool(xb[m23], (y[m23] == 3).astype(int), feature_names=M.B_COLS))
        c = CatBoostClassifier(**M.C_PARAMS, random_seed=seed, iterations=M.C_TREES, **common)
        c.fit(Pool(xc[m56], (y[m56] == 6).astype(int), feature_names=M.C_COLS))
        assert [int(k) for k in a.classes_] == M.LABELS and [int(k) for k in b.classes_] == [0, 1] and [int(k) for k in c.classes_] == [0, 1]
        files = {k: out / f'{k}_s{seed}.cbm' for k in 'ABC'}
        a.save_model(str(files['A'])); b.save_model(str(files['B'])); c.save_model(str(files['C']))
        pa_val.append(a.predict_proba(xa_val)); r3_val.append(b.predict_proba(xb_val)[:, 1]); r6_val.append(c.predict_proba(xc_val)[:, 1])
        models.append({'seed': seed, **{k: {'file': files[k].name, 'trees': m.tree_count_, 'sha256': sha256(files[k])} for k, m in (('A', a), ('B', b), ('C', c))}})
        print(f'[시드 {seed}] 학습 완료 ({time.time() - t0:.0f}초)', flush=True)

    pa, r3, r6 = np.mean(pa_val, axis=0), np.mean(r3_val, axis=0), np.mean(r6_val, axis=0)
    L = np.asarray(M.LABELS)
    pred_a = L[pa.argmax(1)]
    pred_abc = L[M.compose(pa, r3, r6).argmax(1)]
    pred = M.rules.apply(val, pred_abc, use_rule1=use_rule1, use_peak_guard=use_peak_guard, use_stage4_min=use_stage4_min)
    odd = val['experiment_id'].isin(M.EXCLUDE).to_numpy()
    score = lambda p: {'macro_f1': M.macro_f1(y_val, p), 'macro_f1_without_excluded': M.macro_f1(y_val[~odd], p[~odd])}   # noqa: E731
    report = {'A_only': score(pred_a), 'A_B_C': score(pred_abc),
              'A_B_C_rules': {**score(pred), 'accuracy': float(accuracy_score(y_val, pred)), 'rows_changed_by_rules': int((pred != pred_abc).sum()),
                              'per_class_f1': f1_score(y_val, pred, labels=M.LABELS, average=None, zero_division=0).tolist(),
                              'confusion': confusion_matrix(y_val, pred, labels=M.LABELS).tolist()}}
    in_sample = refit
    meta = {
        'name': 'team2_final', 'data': 'train+validation' if in_sample else 'train', 'labels': M.LABELS, 'seeds': M.SEEDS,
        'A_columns': M.A_COLS, 'B_columns': M.B_COLS, 'C_columns': M.C_COLS,
        'A_params': M.A_PARAMS, 'B_params': M.B_PARAMS, 'C_params': M.C_PARAMS,
        'image_feature_version': M.IM.VERSION, 'rules_version': M.rules.VERSION, 'use_rule1': use_rule1, 'use_peak_guard': use_peak_guard, 'use_stage4_min': use_stage4_min,
        'models': models,
        'combine': 'A의 2·3단계 확률 합을 B의 3단계 비율로, 5·6단계 확률 합을 C의 6단계 비율로 나눔. 1·4단계는 A 그대로. 시드 평균 후 가장 높은 단계 → 규칙.',
        'augmentation': '모든 실험을 평균 12초 간격(10·15초)으로 솎은 복사본을 원본과 함께 학습',
        'excluded_experiments': M.EXCLUDE if in_sample else [],
        'n_train_rows': {'original': int(n_orig), 'thinned_copies': int(n_thin), 'B_rows_stage_2_3': int(m23.sum()), 'C_rows_stage_5_6': int(m56.sum())},
        'validation': report,
        'validation_note': ('Validation이 학습에 들어가 있으므로 이 Validation 점수는 성능 추정치가 아니다(학습 데이터 점수).' if in_sample
                            else 'Validation은 학습·선택에 쓰지 않았다(점수 계산만).'),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    (out / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    if not in_sample:
        pd.DataFrame({'sample_id': val['sample_id'], 'label': pred}).to_csv(out / 'val_pred.csv', index=False)
    tag = '(학습에 포함된 데이터의 점수 — 성능 추정치 아님)' if in_sample else '(학습에 쓰지 않은 데이터)'
    print(f'Validation {tag}')
    for key, name in (('A_only', 'A만        '), ('A_B_C', 'A + B + C  '), ('A_B_C_rules', 'A + B + C + 규칙')):
        print(f"  {name}: Macro F1 {report[key]['macro_f1']:.4f} | E_e5bcddc7 제외 {report[key]['macro_f1_without_excluded']:.4f}")
    print('  단계별 F1: ' + ' / '.join(f'{v:.3f}' for v in report['A_B_C_rules']['per_class_f1']) + f" | 규칙이 바꾼 행 {report['A_B_C_rules']['rows_changed_by_rules']}")
    print(f'저장: {out}')


if __name__ == '__main__':
    main()
