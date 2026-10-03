"""
liquidity_size_analysis.py — 「市值/成交值最大的股票，比較不容易是假突破？」
無歷史市值，以『進場訊號日前 60 日平均成交值（收盤×量）在當日全市場的百分位』當規模代理（大型股成交值通常最大）。
依百分位五分位：假訊號率、淨報酬、超額（對全市場同期平均），訓練(<2025)/驗證(>=2025)分開，月分群 CI。
另測 Q5(最大) − Q1(最小)。用法：python liquidity_size_analysis.py --market tw|us
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_cluster_analysis import cluster_stats, boot_diff, baseline_returns, CACHE_DIR, RESULT_DIR
from backtest_portfolio import TW_FEE_BUY, TW_FEE_SELL
from false_signal_analysis import SPLIT, QUICK_DAYS


def load(market):
    o, v = {}, {}
    for f in sorted(CACHE_DIR.glob("*.parquet")):
        is_tw = f.stem.endswith(("_TW", "_TWO"))
        if (market == "tw") != is_tw or f.stem.startswith("_"):
            continue
        try:
            d = pd.read_parquet(f, columns=["Open", "Close", "Volume"])
            d.index = pd.to_datetime(d.index).tz_localize(None)
            o[f.stem] = d["Open"]; v[f.stem] = d["Close"] * d["Volume"]
        except Exception:
            pass
    O = pd.DataFrame(o).sort_index()
    return O.where(O > 0), pd.DataFrame(v).reindex(O.index)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--market", default="tw", choices=["tw", "us"])
    mk = ap.parse_args().market
    pat = "signals_tw_2*.csv" if mk == "tw" else "signals_us_sp500_2*.csv"
    tr = pd.read_csv(sorted(RESULT_DIR.glob(pat))[-1])
    tr = tr[~tr.symbol.str.startswith("_")].copy()
    fee = (TW_FEE_BUY + TW_FEE_SELL) if mk == "tw" else 0.001
    if mk == "us":
        tr["return_pct"] -= 0.1
    O, V = load(mk)
    adv = V.rolling(60, min_periods=40).mean()
    rank = adv.rank(axis=1, pct=True)
    di = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(O.index)}
    cj = {c: j for j, c in enumerate(O.columns)}
    rv = rank.values
    pct = []
    for r in tr.itertuples():
        i, j = di.get(r.entry_date), cj.get(r.symbol.replace(".", "_"))
        pct.append(np.nan if i is None or j is None or i < 1 else rv[i - 1, j])   # 進場訊號日 = 進場前一交易日
    tr["pct"] = pct
    tr["base"] = baseline_returns(tr, O, fee)
    tr = tr.dropna(subset=["pct", "base", "return_pct"]).copy()
    tr["ex"] = tr.return_pct - tr.base
    tr["quick"] = ((tr.days_held <= QUICK_DAYS) & (tr.return_pct < 0)).astype(float)
    tr["month"] = pd.to_datetime(tr.entry_date).dt.to_period("M").astype(str)
    tr["q"] = pd.cut(tr.pct, [0, .2, .4, .6, .8, 1.0], labels=["Q1小", "Q2", "Q3", "Q4", "Q5大"], include_lowest=True)
    print(f"[{mk.upper()}] {len(tr)} 筆；成交值分位五等分（Q5=成交值最大）")
    for label, d in (("全期", tr), ("訓練 2021–2024", tr[tr.entry_date < SPLIT]), ("驗證 2025–2026", tr[tr.entry_date >= SPLIT])):
        print(f"\n=== {label} ===\n  {'分位':<6}{'n':>6}{'假訊號%':>8}{'虧損%':>7}{'均報酬':>8}{'超額[月分群CI]':>22}{'大虧(<-20%)':>12}")
        for g in tr.q.cat.categories:
            x = d[d.q == g]
            if len(x) < 30: continue
            e = cluster_stats(x.ex.values, x.month)
            print(f"  {g:<6}{len(x):>6}{x.quick.mean()*100:>8.1f}{(x.return_pct<0).mean()*100:>7.1f}{x.return_pct.mean():>+7.2f}%  {e['mean']:>+6.2f}[{e['lo']:>+5.2f},{e['hi']:>+5.2f}]{(x.return_pct<-20).mean()*100:>11.1f}%")
        hi, lo = d[d.q == "Q5大"], d[d.q == "Q1小"]
        if len(hi) > 30 and len(lo) > 30:
            for col, nm, sc in (("quick", "假訊號率(pp)", 100), ("ex", "超額", 1), ("return_pct", "均報酬", 1)):
                pe, el, eh, pv = boot_diff(hi, lo, col)
                print(f"  Q5大 − Q1小 {nm}: {pe*sc:+.2f} [{el*sc:+.2f},{eh*sc:+.2f}] p={pv:.3f}")


if __name__ == "__main__":
    main()
