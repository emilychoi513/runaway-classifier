#!/usr/bin/env python
"""추론 스크립트: 입력 CSV → 제출 형식 CSV(sample_id,label).

예) python test.py --input <battery_data_light>/test/test.csv \\
                   --image_root <battery_data_light>/test \\
                   --model weights --output submission.csv \\
                   [--sample_submission <battery_data_light>/sample_submission.csv]

- weights/ 에 저장된 모델만 불러 쓴다. 재학습, 인터넷 접속, 입력 데이터로 통계를 추정하는 일이 없다.
- 입력 CSV에 들어 있는 행만으로 피처를 만든다. 과거 방향 피처와 규칙은 같은 experiment_id의 현재 행과 그 이전 행만 쓴다.
  각 행의 예측은 같은 실험의 이후 행, 다른 실험, 입력 행 순서와 무관하다.
- 이미지는 --image_root 아래에서 매번 새로 읽는다(캐시 없음).
- 출력 행 순서는 입력 CSV 그대로다. --sample_submission 을 주면 그 양식의 순서로 저장한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import pandas as pd

import model as M


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', type=Path, required=True, help='test.csv와 같은 형식의 CSV')
    ap.add_argument('--image_root', type=Path, required=True,
                    help='이미지 폴더의 상위 폴더(예: <battery_data_light>/test). 이미지 폴더 자체(예: .../test/images)를 줘도 같은 결과')
    ap.add_argument('--model', type=Path, default=M.ROOT / 'weights', help='train.py가 저장한 가중치 폴더')
    ap.add_argument('--output', type=Path, default=Path('submission.csv'))
    ap.add_argument('--sample_submission', type=Path, default=None, help='주면 제출 양식의 행 순서로 저장')
    args = ap.parse_args()

    frame = pd.read_csv(args.input)
    pred = pd.Series(M.predict(frame, args.image_root, args.model).astype(int), index=frame['sample_id'].astype(str).to_numpy())
    if args.sample_submission is not None:
        ids = pd.read_csv(args.sample_submission)['sample_id'].astype(str).tolist()
        assert set(ids) == set(pred.index), '입력 CSV와 제출 양식의 sample_id가 다릅니다.'
    else:
        ids = frame['sample_id'].astype(str).tolist()
    out = pd.DataFrame({'sample_id': ids, 'label': pred.loc[ids].to_numpy()})
    assert len(out) == len(frame) and out['label'].isin(M.LABELS).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    counts = out['label'].value_counts().sort_index()
    print(f'{len(out)}행 → {args.output} | 예측 단계별 행 수 ' + ', '.join(f'{k}: {v}' for k, v in counts.items()))


if __name__ == '__main__':
    main()
