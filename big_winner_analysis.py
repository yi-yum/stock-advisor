"""
big_winner_analysis.py — 哪些進場當下可知的特徵，能提高抓到「大波段」(淨報酬 >= 20%) 的機率？
目標與假訊號分析不同：這裡看右尾。特徵（皆為進場訊號日當天可得）：
  CSV 內 rsi/vol_ratio/atr_pct/dist_high/dist_low/price，加上 ret1/5/20/60、乖離20日、相對大盤60日強度(rs60)、
  成交值百分位(liq)。五分位切點用訓練期(<2025)，驗證期(>=2025)套用；Q5−Q1 以進場月分群 bootstrap。
只採信『訓練與驗證同方向且皆顯著』者。用法：python big_winner_analysis.py --market tw|us
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_cluster_analysis import boot_diff, CACHE_DIR, RESULT_DIR
from false_signal_analysis import SPLIT

BIG = 20.0
FEATS = [("rsi", "RSI"), ("vol_ratio", "量比"), ("atr_pct", "ATR%"), ("dist_high", "距52週高"), ("dist_low", "距52週低"),
         ("price", "股價"), ("ret1", "訊號日漲幅"), ("ret5", "近5日漲幅"), ("ret20", "近20日漲幅"), ("ret60", "近60日漲幅"),
         ("ext20", "乖離20日線"), ("rs60", "近60日相對市場強度"), ("liq", "成交值百分位")]


def load(mk):
    c, v = {}, {}
    for f in sorted(CACHE_DIR.glob("*.parquet")):
        if (mk == "tw") != f.stem.endswith(("_TW", "_TWO")) or f.stem.startswith("_"):
            continue
        try:
            d = pd.read_parquet(f, columns=["Close", "Volume"])
            d.index = pd.to_datetime(d.index).tz_localize(None)
            c[f.stem], v[f.stem] = d["Close"], d["Close"] * d["Volume"]
        except Exception:
            pass
    C = pd.DataFrame(c).sort_index(); C = C.where(C > 0)
    return C, pd.DataFrame(v).reindex(C.index)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--market", default="tw", choices=["tw", "us"])
    mk = ap.parse_args().market
    tr = pd.read_csv(sorted(RESULT_DIR.glob("signals_tw_2*.csv" if mk == "tw" else "signals_us_sp500_2*.csv"))[-1])
    tr = tr[~tr.symbol.str.startswith("_")].copy()
    if mk == "us":
        tr["return_pct"] -= 0.1
    C, V = load(mk)
    A = C.values
    sma20 = C.rolling(20, min_periods=15).mean().values
    liq = V.rolling(60, min_periods=40).mean().rank(axis=1, pct=True).values
    mret60 = np.nanmean(A[60:] / A[:-60] - 1, axis=1)
    mret60 = np.concatenate([np.full(60, np.nan), mret60])
    di = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(C.index)}
    cj = {c: j for j, c in enumerate(C.columns)}
    cols = {k: [] for k in ("ret1", "ret5", "ret20", "ret60", "ext20", "rs60", "liq")}
    for r in tr.itertuples():
        i, j = di.get(r.entry_date), cj.get(r.symbol.replace(".", "_"))
        i = i - 1 if i is not None else None
        if i is None or j is None or i < 61:
            for k in cols: cols[k].append(np.nan)
            continue
        c0 = A[i, j]
        cols["ret1"].append((c0 / A[i - 1, j] - 1) * 100); cols["ret5"].append((c0 / A[i - 5, j] - 1) * 100)
        cols["ret20"].append((c0 / A[i - 20, j] - 1) * 100); r60 = c0 / A[i - 60, j] - 1
        cols["ret60"].append(r60 * 100); cols["ext20"].append((c0 / sma20[i, j] - 1) * 100)
        cols["rs60"].append((r60 - mret60[i]) * 100); cols["liq"].append(liq[i, j])
    for k, v in cols.items():
        tr[k] = v
    tr["big"] = (tr.return_pct >= BIG).astype(float)
    tr["month"] = pd.to_datetime(tr.entry_date).dt.to_period("M").astype(str)
    train, test = tr[tr.entry_date < SPLIT], tr[tr.entry_date >= SPLIT]
    print(f"[{mk.upper()}] {len(tr)} 筆；大波段(>= {BIG:.0f}%) 占 {tr.big.mean()*100:.1f}%（訓練 {train.big.mean()*100:.1f} / 驗證 {test.big.mean()*100:.1f}）；"
          f"其平均報酬 {tr[tr.big==1].return_pct.mean():+.1f}%，貢獻總報酬 {tr[tr.big==1].return_pct.sum()/tr.return_pct.sum()*100:.0f}%\n")
    stable = []
    for col, lab in FEATS:
        x = train[col].dropna()
        cuts = np.unique(np.quantile(x, [0, .2, .4, .6, .8, 1]))
        if len(cuts) < 6:
            continue
        res, line = [], []
        for nm, d in (("訓練", train), ("驗證", test)):
            q = pd.cut(d[col], cuts, labels=False, include_lowest=True)
            rates = [d[q == k].big.mean() * 100 if (q == k).sum() >= 30 else np.nan for k in range(5)]
            hi, lo = d[q == 4].dropna(subset=[col]), d[q == 0].dropna(subset=[col])
            pe, el, eh, pv = boot_diff(hi, lo, "big") if len(hi) > 30 and len(lo) > 30 else (np.nan,) * 4
            res.append((pe, pv))
            line.append(f"{nm} 大波段率 Q1→Q5: " + " ".join(f"{r:4.1f}" for r in rates) + f" | Q5−Q1 {pe*100:+.1f}pp p={pv:.3f}")
        print(f"{lab:<14}" + "\n              ".join(line))
        (p1, v1), (p2, v2) = res
        if np.sign(p1) == np.sign(p2) and v1 < 0.05 and v2 < 0.10:
            stable.append((lab, p1 * 100, p2 * 100))
    print("\n=== 訓練/驗證同方向且顯著 ===")
    for lab, a, b in stable:
        print(f"  {lab}: Q5−Q1 大波段率差 訓練 {a:+.1f}pp / 驗證 {b:+.1f}pp")
    if not stable:
        print("  （無）")


if __name__ == "__main__":
    main()
