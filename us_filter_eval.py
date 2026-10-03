"""
us_filter_eval.py — 把台股找到的假訊號過濾規則拿到美股（S&P500 回測）做樣本外複驗
用法：python us_filter_eval.py
流程與 false_signal_analysis.py / filter_eval.py 相同：切點取美股訓練期（進場<2025）分位，
驗證期為 2025–2026；成本採 tracker 的來回 0.1%（回測 CSV 本身未扣費）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_cluster_analysis import cluster_stats, boot_diff, baseline_returns, CACHE_DIR, RESULT_DIR
from backtest_risk_analysis import nw_alpha, port_stats
from false_signal_analysis import add_features, SPLIT, QUICK_DAYS

FEE_LEG, ANN = 0.0005, 252


def load_OC_us():
    o, c = {}, {}
    for f in sorted(CACHE_DIR.glob("*.parquet")):
        if f.stem.endswith(("_TW", "_TWO")) or f.stem.startswith("_"):
            continue
        try:
            d = pd.read_parquet(f, columns=["Open", "Close"])
            d.index = pd.to_datetime(d.index).tz_localize(None)
            o[f.stem], c[f.stem] = d["Open"], d["Close"]
        except Exception:
            pass
    O = pd.DataFrame(o).sort_index()
    C = pd.DataFrame(c).reindex(O.index)
    return O.where(O > 0), C.where(C > 0)


def build_portfolio(trades, D, dates, cols):
    n = D.shape[0]
    ssum, scnt = np.zeros(n), np.zeros(n)
    for r in trades.itertuples():
        ie, ix, j = dates.get(r.entry_date), dates.get(r.exit_date), cols.get(r.symbol.replace(".", "_"))
        if ie is None or ix is None or j is None or ix <= ie:
            continue
        seg = D[ie:ix, j].copy()
        seg[0] -= FEE_LEG
        seg[-1] -= FEE_LEG
        ssum[ie:ix] += seg
        scnt[ie:ix] += 1
    return np.where(scnt > 0, ssum / np.maximum(scnt, 1), 0.0)


def main():
    tr = pd.read_csv(sorted(RESULT_DIR.glob("signals_us_sp500_2*.csv"))[-1])
    tr = tr[~tr.symbol.str.startswith("_")].copy()
    tr["return_pct"] -= 2 * FEE_LEG * 100
    O, C = load_OC_us()
    print(f"[資料] 美股回測 {len(tr)} 筆；全市場矩陣 {O.shape[1]} 檔")
    tr["base"] = baseline_returns(tr, O, 2 * FEE_LEG)
    tr = add_features(tr, C).dropna(subset=["base", "return_pct", "dist_high", "vol_ratio", "ret1"]).copy()
    tr["excess"] = tr["return_pct"] - tr["base"]
    tr["quick"] = ((tr["days_held"] <= QUICK_DAYS) & (tr["return_pct"] < 0)).astype(float)
    tr["month"] = pd.to_datetime(tr["entry_date"]).dt.to_period("M").astype(str)
    is_train = (tr.entry_date < SPLIT).values
    train, test = tr[is_train], tr[~is_train]
    print(f"訓練 {len(train)} / 驗證 {len(test)}；假訊號率 {tr.quick.mean()*100:.1f}%（訓練 {train.quick.mean()*100:.1f} / 驗證 {test.quick.mean()*100:.1f}）；"
          f"假訊號均報酬 {tr[tr.quick==1].return_pct.mean():+.2f}%，其餘 {tr[tr.quick==0].return_pct.mean():+.2f}%\n")

    print("── 特徵五分位 Q5 vs Q1（切點取訓練期）──")
    for col, label in (("vol_ratio", "量比"), ("ret1", "訊號日漲幅"), ("dist_high", "距52週高(越大越近)"),
                       ("rsi", "RSI"), ("atr_pct", "ATR%"), ("ret20", "近20日漲幅")):
        cuts = np.unique(np.nanquantile(train[col], [0, .2, .4, .6, .8, 1]))
        if len(cuts) < 6:
            continue
        out = []
        for nm, d in (("訓練", train), ("驗證", test)):
            q = pd.cut(d[col], cuts, labels=False, include_lowest=True)
            hi, lo = d[q == 4], d[q == 0]
            if len(hi) < 30 or len(lo) < 30:
                out.append(f"{nm} n/a"); continue
            pe, el, eh, pv = boot_diff(hi, lo, "excess")
            pq, ql, qh, qp = boot_diff(hi, lo, "quick")
            out.append(f"{nm} 假訊號率 Q1 {lo.quick.mean()*100:.1f}% → Q5 {hi.quick.mean()*100:.1f}% (差 {pq*100:+.1f}pp, p={qp:.3f})；"
                       f"超額差 {pe:+.2f} (p={pv:.3f})")
        print(f"  {label}\n    " + "\n    ".join(out))

    q = lambda col, p: float(np.quantile(train[col], p))
    cut_dh60, cut_vr, cut_r1 = q("dist_high", .6), q("vol_ratio", .8), q("ret1", .8)
    print(f"\n[切點] 距高 >{cut_dh60:.3f}；量比 <={cut_vr:.2f}；訊號日漲幅 <={cut_r1:.2f}%")
    rules = {
        "全部(基準)": np.ones(len(tr), bool),
        "R1 距52週高 前40%": (tr.dist_high > cut_dh60).values,
        "R2 量比不暴增": (tr.vol_ratio <= cut_vr).values,
        "R3 訊號日漲幅不過大": (tr.ret1 <= cut_r1).values,
        "R2+R3": ((tr.vol_ratio <= cut_vr) & (tr.ret1 <= cut_r1)).values,
        "R1+R2+R3": ((tr.dist_high > cut_dh60) & (tr.vol_ratio <= cut_vr) & (tr.ret1 <= cut_r1)).values,
    }
    dates = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(O.index)}
    cols = {c: j for j, c in enumerate(O.columns)}
    with np.errstate(all="ignore"):
        Draw = O.values[1:] / O.values[:-1] - 1
    D = np.where(np.isfinite(Draw), Draw, 0.0)
    uni = np.nanmean(Draw, axis=1)
    uni = np.where(np.isfinite(uni), uni, 0.0)
    ddates = O.index[1:]
    series = {k: build_portfolio(tr[m], D, dates, cols) for k, m in rules.items()}

    for label, ptrain in (("訓練期 2021–2024", True), ("驗證期（樣本外）2025–2026", False)):
        dm = (ddates < pd.Timestamp(SPLIT)) if ptrain else (ddates >= pd.Timestamp(SPLIT))
        sel = is_train if ptrain else ~is_train
        sub = tr[sel]
        print(f"\n=== {label} ===")
        print(f"  {'規則':<18}{'保留':>6}{'假訊號%':>8}{'虧損%':>7}{'均報酬':>8}{'超額[月分群CI]':>22}{'CAGR':>8}{'Sharpe':>7}{'MDD':>8}{'α/年(t)':>14}")
        for k, m in rules.items():
            g = sub[m[sel]]
            ex = cluster_stats(g["excess"].values, g["month"])
            ps = port_stats(series[k][dm])
            a, _, ta = nw_alpha(series[k][dm], uni[dm])
            print(f"  {k:<18}{len(g)/len(sub)*100:>5.0f}%{g.quick.mean()*100:>8.1f}{(g.return_pct<0).mean()*100:>7.1f}{g.return_pct.mean():>+7.2f}%"
                  f"  {ex['mean']:>+6.2f}[{ex['lo']:>+5.2f},{ex['hi']:>+5.2f}]{ps['cagr']:>7.1f}%{ps['sharpe']:>7.2f}{ps['maxdd']:>7.1f}%{a*ANN*100:>+8.1f}%({ta:>+5.2f})")
        pu = port_stats(uni[dm])
        print(f"  {'[基準]全市場等權':<18}{'':>6}{'':>8}{'':>7}{'':>8}{'':>22}{pu['cagr']:>7.1f}%{pu['sharpe']:>7.2f}{pu['maxdd']:>7.1f}%")


if __name__ == "__main__":
    main()
