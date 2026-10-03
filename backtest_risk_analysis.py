"""
backtest_risk_analysis.py — 三重 SuperTrend 回測的「風險面」檢驗
=================================================================
backtest_cluster_analysis.py 顯示：相對「同期持有全市場」，訊號的平均超額報酬約為 0。
本腳本檢驗另一種可能的價值——策略是否靠移動停損/出場降低了風險：

  A. 單筆交易風險：每筆交易 vs 「同一個持有期間、隨機抽 K 檔同市場股票持有」的對照組
     （報酬分布、下行偏差、虧損機率、CVaR、持有期間最大不利波動 MAE）。
     以月分群算「策略 − 對照」差異的 95% 信賴區間。
  B. 組合層級：把所有進行中的交易每日等權重組成投資組合，與全市場等權比較
     （年化報酬/波動、Sharpe、最大回撤、Beta、上/下行捕捉率、年化 alpha）。

用法：python backtest_risk_analysis.py --market tw   /   --market us
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_cluster_analysis import cluster_stats, CACHE_DIR, RESULT_DIR
from backtest_portfolio import TW_FEE_BUY, TW_FEE_SELL

K_CONTROL = 10
ANN = 252


def load_matrices(market: str):
    files = sorted(CACHE_DIR.glob("*.parquet"))
    tw = [f for f in files if f.stem.endswith("_TW") or f.stem.endswith("_TWO")]
    sel = tw if market == "tw" else [f for f in files if f not in tw]
    o, lo = {}, {}
    for f in sel:
        try:
            d = pd.read_parquet(f, columns=["Open", "Low"])
            d.index = pd.to_datetime(d.index).tz_localize(None)
            o[f.stem], lo[f.stem] = d["Open"], d["Low"]
        except Exception:
            pass
    O = pd.DataFrame(o).sort_index()
    L = pd.DataFrame(lo).reindex(O.index)
    return O.where(O > 0), L.where(L > 0)


def dist_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    k = max(1, int(len(x) * 0.05))
    return {
        "mean": x.mean(), "median": np.median(x), "std": x.std(ddof=1),
        "down_dev": np.sqrt(np.mean(np.minimum(x, 0) ** 2)),
        "p_loss": (x < 0).mean() * 100, "p_lt10": (x < -10).mean() * 100, "p_lt20": (x < -20).mean() * 100,
        "cvar5": np.sort(x)[:k].mean(), "p1": np.percentile(x, 1), "p5": np.percentile(x, 5),
        "p95": np.percentile(x, 95), "worst": x.min(),
    }


def show_dist(label: str, s: dict, c: dict):
    rows = [("平均", "mean", "%"), ("中位數", "median", "%"), ("標準差", "std", "%"), ("下行偏差", "down_dev", "%"),
            ("虧損機率", "p_loss", "%"), ("跌幅>10% 機率", "p_lt10", "%"), ("跌幅>20% 機率", "p_lt20", "%"),
            ("最差 5% 平均(CVaR5)", "cvar5", "%"), ("第 1 百分位", "p1", "%"), ("第 5 百分位", "p5", "%"),
            ("第 95 百分位", "p95", "%"), ("最差單筆", "worst", "%")]
    print(f"  {label:<22}{'策略':>10}{'對照(隨機持有)':>16}")
    for name, key, _ in rows:
        print(f"  {name:<22}{s[key]:>9.2f}%{c[key]:>15.2f}%")


def nw_alpha(y: np.ndarray, x: np.ndarray, lags: int = 5):
    """y = a + b x 的 OLS，Newey-West 標準誤。回傳 a, b, t(a)。"""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    n = len(y)
    xe = X * e[:, None]
    S = xe.T @ xe / n
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = xe[l:].T @ xe[:-l] / n
        S += w * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X / n)
    V = XtX_inv @ S @ XtX_inv / n
    return beta[0], beta[1], beta[0] / np.sqrt(V[0, 0])


def port_stats(r: np.ndarray) -> dict:
    eq = np.cumprod(1 + r)
    dd = eq / np.maximum.accumulate(eq) - 1
    yrs = len(r) / ANN
    cagr = eq[-1] ** (1 / yrs) - 1
    vol = r.std(ddof=1) * np.sqrt(ANN)
    m = pd.Series(r).groupby(np.arange(len(r)) // 21).apply(lambda s: (1 + s).prod() - 1)
    return {"cagr": cagr * 100, "vol": vol * 100, "sharpe": (r.mean() * ANN) / vol if vol else np.nan,
            "maxdd": dd.min() * 100, "calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
            "worst_day": r.min() * 100, "worst_month": m.min() * 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="tw", choices=["tw", "us"])
    mk = ap.parse_args().market

    pat = "signals_tw_2*.csv" if mk == "tw" else "signals_us_sp500_2*.csv"
    f = sorted(RESULT_DIR.glob(pat))[-1]
    tr = pd.read_csv(f)
    O, L = load_matrices(mk)
    fee_buy, fee_sell = (TW_FEE_BUY, TW_FEE_SELL) if mk == "tw" else (0.0, 0.0)
    fee_rt = fee_buy + fee_sell
    print(f"[交易] {f.name}：{len(tr)} 筆；[價格] {O.shape[1]} 檔 × {O.shape[0]} 日")

    dates = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(O.index)}
    colmap = {c: j for j, c in enumerate(O.columns)}
    oa, la = O.values, L.values

    # ── A. 單筆風險 vs 隨機持有對照 ─────────────────────────────
    rng = np.random.default_rng(7)
    rec = []
    for r in tr.itertuples():
        ie, ix = dates.get(r.entry_date), dates.get(r.exit_date)
        j = colmap.get(r.symbol.replace(".", "_"))
        if ie is None or ix is None or ix <= ie or j is None or not np.isfinite(oa[ie, j]):
            continue
        valid = np.flatnonzero(np.isfinite(oa[ie]) & np.isfinite(oa[ix]))
        if len(valid) < K_CONTROL:
            continue
        cols = rng.choice(valid, K_CONTROL, replace=False)
        lows = la[ie:ix][:, [j] + list(cols)]
        with np.errstate(all="ignore"):
            mae = np.nanmin(lows, axis=0) / oa[ie, [j] + list(cols)] - 1
        cret = (oa[ix, cols] / oa[ie, cols] - 1 - fee_rt) * 100
        if not np.isfinite(mae[0]) or not np.isfinite(mae[1:]).all():
            continue
        rec.append((r.signal_date, r.return_pct, mae[0] * 100, cret, mae[1:] * 100))
    sig = pd.Series([x[0] for x in rec])
    month = pd.to_datetime(sig).dt.to_period("M").astype(str)
    s_ret = np.array([x[1] for x in rec]); s_mae = np.array([x[2] for x in rec])
    c_ret = np.concatenate([x[3] for x in rec]); c_mae = np.concatenate([x[4] for x in rec])
    print(f"\n=== A. 單筆交易風險：策略 {len(rec)} 筆 vs 同窗隨機持有對照 {len(c_ret)} 筆（每筆 {K_CONTROL} 檔）===")
    print("-- 持有期報酬分布")
    show_dist("報酬", dist_stats(s_ret), dist_stats(c_ret))
    print("-- 持有期間最大不利波動 MAE（期間最低價相對進場開盤）")
    show_dist("MAE", dist_stats(s_mae), dist_stats(c_mae))

    print("\n-- 配對差異（策略 − 同窗對照平均），月分群 95% CI：")
    pairs = [
        ("持有期報酬平均", s_ret, np.array([x[3].mean() for x in rec])),
        ("虧損機率 (報酬<0)", (s_ret < 0) * 100.0, np.array([(x[3] < 0).mean() * 100 for x in rec])),
        ("大虧機率 (報酬<-10%)", (s_ret < -10) * 100.0, np.array([(x[3] < -10).mean() * 100 for x in rec])),
        ("大虧機率 (報酬<-20%)", (s_ret < -20) * 100.0, np.array([(x[3] < -20).mean() * 100 for x in rec])),
        ("深度回撤機率 (MAE<-10%)", (s_mae < -10) * 100.0, np.array([(x[4] < -10).mean() * 100 for x in rec])),
        ("深度回撤機率 (MAE<-20%)", (s_mae < -20) * 100.0, np.array([(x[4] < -20).mean() * 100 for x in rec])),
        ("平均 MAE", s_mae, np.array([x[4].mean() for x in rec])),
        ("下行損失(min(報酬,0))", np.minimum(s_ret, 0), np.array([np.minimum(x[3], 0).mean() for x in rec])),
    ]
    for name, s_, c_ in pairs:
        cs = cluster_stats(s_ - c_, month)
        print(f"  {name:<24} 差 {cs['mean']:+7.2f}  95%CI[{cs['lo']:+7.2f}, {cs['hi']:+7.2f}]  t={cs['t']:+5.2f}")

    # ── B. 組合層級（所有進行中交易每日等權重）────────────────
    n_days = len(O.index)
    with np.errstate(all="ignore"):
        Draw = oa[1:] / oa[:-1] - 1
    D = np.where(np.isfinite(Draw), Draw, 0.0)
    uni_all = np.nanmean(Draw, axis=1)

    def build(fb: float, fs: float):
        ssum, scnt = np.zeros(n_days - 1), np.zeros(n_days - 1)
        for r in tr.itertuples():
            ie, ix = dates.get(r.entry_date), dates.get(r.exit_date)
            j = colmap.get(r.symbol.replace(".", "_"))
            if ie is None or ix is None or ix <= ie or j is None:
                continue
            seg = D[ie:ix, j].copy()
            seg[0] -= fb
            seg[-1] -= fs
            ssum[ie:ix] += seg
            scnt[ie:ix] += 1
        return ssum, scnt

    ssum, scnt = build(fee_buy, fee_sell)
    first = int(np.argmax(scnt > 0))
    active = scnt[first:] > 0
    uni = np.where(np.isfinite(uni_all[first:]), uni_all[first:], 0.0)

    def strat_series(ssum, scnt):
        return np.where(scnt[first:] > 0, ssum[first:] / np.maximum(scnt[first:], 1), 0.0)

    print(f"\n=== B. 組合層級：進行中交易每日等權重 vs 全市場等權（{len(uni)} 個交易日）===")
    print(f"  有部位的天數比例 {active.mean()*100:.0f}%；同時持有檔數 平均 {scnt[first:][active].mean():.0f}、最多 {int(scnt.max())}")
    pu = port_stats(uni)
    gs, gc = build(0.0, 0.0)
    cols = [("策略(扣成本)", strat_series(ssum, scnt)), ("策略(不扣成本)", strat_series(gs, gc)), ("全市場等權", uni)]
    stats = [port_stats(x) for _, x in cols]
    print(f"  {'':<18}" + "".join(f"{n:>14}" for n, _ in cols))
    for name, key, unit in (("年化報酬", "cagr", "%"), ("年化波動", "vol", "%"), ("Sharpe(無風險=0)", "sharpe", ""),
                            ("最大回撤", "maxdd", "%"), ("Calmar", "calmar", ""), ("最差單日", "worst_day", "%"),
                            ("最差 21 日區間", "worst_month", "%")):
        print(f"  {name:<20}" + "".join(f"{st_[key]:>13.2f}{unit or ' '}" for st_ in stats))
    up, dn = uni > 0, uni < 0
    for name, series in cols[:2]:
        a, b, ta = nw_alpha(series, uni)
        print(f"  [{name}] Beta {b:.2f}；年化 alpha {a*ANN*100:+.1f}%（Newey-West t={ta:+.2f}）；"
              f"上行捕捉 {series[up].mean()/uni[up].mean()*100:.0f}%、下行捕捉 {series[dn].mean()/uni[dn].mean()*100:.0f}%")
    turn = len(tr) / max(scnt[first:][active].mean(), 1) / (len(uni) / ANN)
    print(f"  平均每個持倉位置每年換手 {turn:.1f} 次（每次來回成本 {fee_rt*100:.3f}%，約 {turn*fee_rt*100:.1f}%/年）")
    ser = cols[0][1]
    yr = pd.Series(ser, index=O.index[1:][first:]).groupby(lambda d: d.year).apply(lambda s_: ((1 + s_).prod() - 1) * 100)
    yu = pd.Series(uni, index=O.index[1:][first:]).groupby(lambda d: d.year).apply(lambda s_: ((1 + s_).prod() - 1) * 100)
    print("  逐年報酬(扣成本)：" + "  ".join(f"{y}: 策略 {yr[y]:+.0f}% / 市場 {yu[y]:+.0f}%" for y in yr.index))
    print("\n  注意：組合為「每日再平衡的等權重」近似，未模擬資金與持倉數限制；全市場基準視為持有不動、未計成本。")


if __name__ == "__main__":
    main()
