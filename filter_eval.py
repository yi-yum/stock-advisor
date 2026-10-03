"""
filter_eval.py — 假訊號過濾規則的樣本外驗證（台股三重 SuperTrend 回測）
=====================================================================
false_signal_analysis.py 找出訓練/驗證期方向一致的特徵：
  · 量比暴增、訊號日大漲 → 假訊號（出場快且虧損）比例明顯較高
  · 距 52 週高點越近 → 超額報酬越好
本腳本把它們變成過濾規則（切點一律用訓練期 2021–2024 的分位決定），
在訓練期與驗證期（2025–2026）比較「保留的交易」vs「全部交易」：
  交易層級：假訊號率、虧損率、平均淨報酬、超額報酬（月分群 95%CI）
  組合層級：保留交易每日等權重（扣台股成本）的 CAGR、Sharpe、最大回撤與對全市場 alpha

用法：python filter_eval.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_cluster_analysis import cluster_stats, baseline_returns, RESULT_DIR
from backtest_portfolio import TW_FEE_BUY, TW_FEE_SELL
from backtest_risk_analysis import nw_alpha, port_stats
from false_signal_analysis import load_OC, add_features, SPLIT, QUICK_DAYS

ANN = 252


def build_portfolio(trades: pd.DataFrame, O: pd.DataFrame, D: np.ndarray, dates: dict, cols: dict):
    n = len(O.index)
    ssum, scnt = np.zeros(n - 1), np.zeros(n - 1)
    for r in trades.itertuples():
        ie, ix, j = dates.get(r.entry_date), dates.get(r.exit_date), cols.get(r.symbol.replace(".", "_"))
        if ie is None or ix is None or j is None or ix <= ie:
            continue
        seg = D[ie:ix, j].copy()
        seg[0] -= TW_FEE_BUY
        seg[-1] -= TW_FEE_SELL
        ssum[ie:ix] += seg
        scnt[ie:ix] += 1
    return np.where(scnt > 0, ssum / np.maximum(scnt, 1), 0.0), scnt


def main():
    tr = pd.read_csv(sorted(RESULT_DIR.glob("signals_tw_2*.csv"))[-1])
    O, C = load_OC()
    tr["base"] = baseline_returns(tr, O, TW_FEE_BUY + TW_FEE_SELL)
    tr = add_features(tr, C).dropna(subset=["base", "return_pct", "dist_high", "vol_ratio", "ret1"]).copy()
    tr["excess"] = tr["return_pct"] - tr["base"]
    tr["quick"] = ((tr["days_held"] <= QUICK_DAYS) & (tr["return_pct"] < 0)).astype(float)
    tr["month"] = pd.to_datetime(tr["entry_date"]).dt.to_period("M").astype(str)
    is_train = tr.entry_date < SPLIT
    train = tr[is_train]

    q = lambda col, p: float(np.quantile(train[col], p))
    cut_dh60, cut_dh80 = q("dist_high", .6), q("dist_high", .8)
    cut_vr, cut_r1 = q("vol_ratio", .8), q("ret1", .8)
    print(f"[切點，來自訓練期] 距52週高 前40%: > {cut_dh60:.3f}、前20%: > {cut_dh80:.3f}（比例，-0.1 = 低於高點 10%）；"
          f"量比 <= {cut_vr:.2f}；訊號日漲幅 <= {cut_r1:.1f}%")

    rules = {
        "全部(基準)":                         np.ones(len(tr), bool),
        "R1 距52週高 前40%":                  (tr.dist_high > cut_dh60).values,
        "R2 量比不暴增":                      (tr.vol_ratio <= cut_vr).values,
        "R3 訊號日漲幅不過大":                (tr.ret1 <= cut_r1).values,
        "R2+R3（只濾假訊號）":                ((tr.vol_ratio <= cut_vr) & (tr.ret1 <= cut_r1)).values,
        "R1+R2+R3":                           ((tr.dist_high > cut_dh60) & (tr.vol_ratio <= cut_vr) & (tr.ret1 <= cut_r1)).values,
        "R1(前20%)+R2+R3":                    ((tr.dist_high > cut_dh80) & (tr.vol_ratio <= cut_vr) & (tr.ret1 <= cut_r1)).values,
    }

    dates = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(O.index)}
    cols = {c: j for j, c in enumerate(O.columns)}
    with np.errstate(all="ignore"):
        Draw = O.values[1:] / O.values[:-1] - 1
    D = np.where(np.isfinite(Draw), Draw, 0.0)
    uni = np.nanmean(Draw, axis=1)
    uni = np.where(np.isfinite(uni), uni, 0.0)
    ddates = O.index[1:]

    series = {name: build_portfolio(tr[m], O, D, dates, cols) for name, m in rules.items()}

    for label, period_train in (("訓練期（切點在此決定）2021–2024", True), ("驗證期（樣本外）2025–2026", False)):
        dm = (ddates < pd.Timestamp(SPLIT)) if period_train else (ddates >= pd.Timestamp(SPLIT))
        sub_all = tr[is_train] if period_train else tr[~is_train]
        print(f"\n=== {label} ===")
        print(f"  {'規則':<22}{'保留':>7}{'假訊號%':>8}{'虧損%':>7}{'均報酬':>8}{'超額[月分群CI]':>20}{'CAGR':>8}{'Sharpe':>7}{'最大回撤':>9}{'α/年(t)':>13}")
        pu = port_stats(uni[dm])
        for name, m in rules.items():
            g = sub_all[m[(tr.entry_date < SPLIT).values] if period_train else m[(tr.entry_date >= SPLIT).values]]
            ex = cluster_stats(g["excess"].values, g["month"])
            ser, cnt = series[name]
            ps = port_stats(ser[dm])
            a, _, ta = nw_alpha(ser[dm], uni[dm])
            share = len(g) / len(sub_all) * 100
            print(f"  {name:<22}{share:>6.0f}%{g.quick.mean()*100:>8.1f}{(g.return_pct<0).mean()*100:>7.1f}{g.return_pct.mean():>+7.2f}%"
                  f"  {ex['mean']:>+6.2f}[{ex['lo']:>+5.2f},{ex['hi']:>+5.2f}]{ps['cagr']:>7.1f}%{ps['sharpe']:>7.2f}{ps['maxdd']:>8.1f}%"
                  f"{a*ANN*100:>+8.1f}%({ta:>+5.2f})")
        print(f"  {'[基準]全市場等權':<22}{'':>7}{'':>8}{'':>7}{'':>8}{'':>20}{pu['cagr']:>7.1f}%{pu['sharpe']:>7.2f}{pu['maxdd']:>8.1f}%")
    print("\n  組合為每日再平衡的等權重近似、已扣台股成本；保留比例愈低，同時持有檔數愈少，波動愈大。")


if __name__ == "__main__":
    main()
