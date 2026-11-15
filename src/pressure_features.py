"""압력(meas_p) 변곡점 피처 48개.

배경: Train·Validation 97개 실험에서 3단계 시작 시점은 meas_p가 그때까지의 추세에서 벗어나기 시작하는 시점
(완만한 상승의 시작, 또는 한 번에 뛰거나 떨어지는 지점)과 맞물려 있고, 압력 변화가 뚜렷하지 않은 실험은
meas_t1이 50℃에 닿는 시점에 3단계가 시작된다. 이 파일은 그 "추세 이탈"을 과거 방향으로만 계산한다.

규칙
- 같은 experiment_id 안에서 현재 행과 그 이전 행만 쓴다. 미래 행·실험 전체 통계는 쓰지 않는다.
- 창 길이는 모두 초 단위(relative_time_s 기준)라서 관측 간격이 달라도 같은 의미다.
- 학습하는 값이 없다(고정 계산식). 입력 행 순서와 무관하게 같은 값이 나온다.
"""
import numpy as np
import pandas as pd


def _lag(t, x, s):
    """각 행에 대해 시각이 (t - s) 이하인 행 중 가장 최근 행의 x. 없으면 NaN. t는 오름차순."""
    j = np.searchsorted(t, t - s, side='right') - 1
    out = np.full(len(t), np.nan); ok = j >= 0; out[ok] = x[j[ok]]; return out


def _trail_fit(t, x, W, G, min_n=4):
    """각 행 i: 시각이 [t_i-W, t_i-G]인 과거 행으로 직선을 맞춰 t_i로 연장한 값, 잔차 표준편차, 평균."""
    n = len(t)
    S = lambda a: np.r_[0.0, np.cumsum(a)]
    St, Sx, Stt, Stx, Sxx = S(t), S(x), S(t * t), S(t * x), S(x * x)
    lo = np.searchsorted(t, t - W, side='left'); hi = np.searchsorted(t, t - G, side='right')
    hi = np.minimum(hi, np.arange(n))            # 현재 행은 기준 구간에 넣지 않음
    k = (hi - lo).astype(float)
    pred = np.full(n, np.nan); sd = np.full(n, np.nan); mean = np.full(n, np.nan)
    ok = k >= 1
    rng = lambda P: P[hi] - P[lo]
    st, sx, stt, stx, sxx = rng(St), rng(Sx), rng(Stt), rng(Stx), rng(Sxx)
    mean[ok] = sx[ok] / k[ok]
    # t를 현재 시각 기준으로 옮김: u = t - t_i
    su = st - k * t; suu = stt - 2 * t * st + k * t * t; sux = stx - t * sx
    den = k * suu - su * su
    good = (k >= min_n) & (den > 1e-9)
    slope = np.where(good, (k * sux - su * sx) / np.where(good, den, 1), 0.0)
    icpt = np.where(k > 0, (sx - slope * su) / np.where(k > 0, k, 1), np.nan)
    pred[ok] = icpt[ok]
    sse = sxx - icpt * sx - slope * sux
    g2 = good & (k >= 3)
    sd[g2] = np.sqrt(np.maximum(sse[g2], 0) / (k[g2] - 2))
    return pred, sd, mean, slope


def _since_first(t, cond):
    """cond가 처음 참이 된 시각부터 지난 초. 그 전 행은 NaN."""
    out = np.full(len(t), np.nan); h = np.flatnonzero(cond)
    if len(h): out[h[0]:] = t[h[0]:] - t[h[0]]
    return out


def _one(t, p, t1):
    """한 실험(시간순)의 압력 피처. t: 시각(초), p: meas_p, t1: meas_t1."""
    n = len(t); F = {}
    reached45 = np.fmax.accumulate(t1) >= 45.0; reached50 = np.fmax.accumulate(t1) >= 50.0
    i45 = np.flatnonzero(reached45); i50 = np.flatnonzero(reached50)
    floor = lambda m: 0.004 * np.abs(np.nan_to_num(m)) + 0.02
    for W, G in ((120, 20), (300, 30)):
        pred, sd, mean, slope = _trail_fit(t, p, W, G)
        sde = np.fmax(np.nan_to_num(sd, nan=0.0), floor(mean))
        dev = p - pred; z = np.clip(dev / sde, -60, 60)
        F[f'p_dev{W}'] = dev; F[f'p_z{W}'] = z; F[f'p_sd{W}'] = sd
        F[f'p_rdev{W}'] = dev / np.fmax(np.abs(mean), 0.5)
        F[f'p_mdev{W}'] = p - mean
        F[f'p_base_slope{W}'] = slope
        za = np.where(np.isfinite(z), z, 0.0)
        F[f'p_z{W}_cummax'] = np.maximum.accumulate(za); F[f'p_z{W}_cummin'] = np.minimum.accumulate(za)
        F[f'p_absz{W}_cummax'] = np.maximum.accumulate(np.abs(za))
        # T1이 45℃에 닿은 뒤부터만 센 값
        zb = np.where(reached45, np.abs(za), 0.0); m45 = np.maximum.accumulate(zb)
        F[f'p_absz{W}_cummax_since45'] = np.where(reached45, m45, np.nan)
        for thr in (4, 8):
            F[f'p_s_since_absz{W}_gt{thr}'] = _since_first(t, reached45 & (np.abs(za) > thr))
        if W == 300:
            for s in (15, 30, 60, 120):
                d = p - _lag(t, p, s); F[f'p_d{s}'] = d; zs = np.clip(d / sde, -60, 60); F[f'p_zd{s}'] = zs
                if s in (30, 60):
                    zz = np.where(np.isfinite(zs), zs, 0.0)
                    F[f'p_zd{s}_cummax'] = np.maximum.accumulate(zz); F[f'p_zd{s}_cummin'] = np.minimum.accumulate(zz)
            prev = np.r_[np.nan, p[:-1]]; zst = np.clip(np.abs(p - prev) / sde, 0, 60)
            F['p_zstep_cummax'] = np.maximum.accumulate(np.where(np.isfinite(zst), zst, 0.0))
            sde300 = sde
    first = p[np.flatnonzero(np.isfinite(p))[0]] if np.isfinite(p).any() else np.nan
    F['p_ratio_start'] = p / first if first and np.isfinite(first) and abs(first) > 1e-6 else np.full(n, np.nan)
    F['p_log'] = np.log1p(np.clip(p, 0, None))
    for nm, idx in (('45', i45), ('50', i50)):
        a = np.full(n, np.nan); b = np.full(n, np.nan); c = np.full(n, np.nan); r = np.full(n, np.nan)
        if len(idx):
            k = idx[0]; ref = p[k]; s0 = sde300[k]
            a[k:] = p[k:] - ref; b[k:] = np.clip((p[k:] - ref) / s0, -200, 200)
            c[k:] = np.clip(np.maximum.accumulate(np.abs(p[k:] - ref)) / s0, 0, 200)
            r[k:] = np.clip((np.maximum.accumulate(p[k:]) - np.minimum.accumulate(p[k:])) / s0, 0, 400)
        F[f'p_chg_since{nm}'] = a; F[f'p_zchg_since{nm}'] = b; F[f'p_zabschg_cummax_since{nm}'] = c; F[f'p_zrange_since{nm}'] = r
    F['past_s_since_t1_50'] = _since_first(t, reached50)
    return F


def p_features(frame: pd.DataFrame) -> pd.DataFrame:
    """압력 피처 표. 실험별로 시간순 정렬해 계산하고 입력 행 순서로 돌려준다."""
    n = len(frame)
    t_all = pd.to_numeric(frame['relative_time_s'], errors='coerce').to_numpy(float)
    p_all = pd.to_numeric(frame['meas_p'], errors='coerce').to_numpy(float)
    t1_all = pd.to_numeric(frame['meas_t1'], errors='coerce').to_numpy(float)
    exp = frame['experiment_id'].astype(str).to_numpy()
    order = np.lexsort((np.arange(n), t_all, exp)); es = exp[order]
    starts = np.flatnonzero(np.r_[True, es[1:] != es[:-1]]); ends = np.r_[starts[1:], n]
    out = None
    for s, e in zip(starts, ends):
        idx = order[s:e]
        F = _one(t_all[idx], p_all[idx], t1_all[idx])
        if out is None: out = {k: np.full(n, np.nan) for k in F}
        for k, v in F.items(): out[k][idx] = v
    return pd.DataFrame(out, index=frame.index)


# 피처 이름 목록. 작은 가짜 입력으로 한 번 계산해 얻는다
P_COLS = list(p_features(pd.DataFrame({"experiment_id": ["e"] * 3, "relative_time_s": [0, 5, 10], "meas_p": [1.0, 1.0, 1.0], "meas_t1": [20.0, 20.0, 20.0]})).columns)
