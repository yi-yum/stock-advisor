"""
false_signal_analysis.py — 哪些特徵能辨識「假訊號」？（台股三重 SuperTrend 回測）
===================================================================
假訊號定義：出場得很快且虧損（持有 <=14 個日曆天且淨報酬 <0），即「洗盤」。
對每個訊號當天可得的特徵（RSI、量比、波動、距 52 週高低點、訊號前漲幅/乖離…）：
  1. 以訓練期（進場 <2025）決定五分位切點，套用到驗證期（進場 >=2025）
  2. 比較各分位的：假訊號率、虧損率、平均淨報酬、超額報酬（扣同期全市場平均）
  3. 檢查「最高分位 − 最低分位」的差異在訓練/驗證期是否同方向且顯著
     （月分群 bootstrap，因同月訊號互相關聯）
特徵多、分位多，只看「訓練與驗證一致且顯著」的，單一期的顯著不採信。

用法：python false_signal_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_cluster_analysis import cluster_stats, boot_diff, baseline_returns, CACHE_DIR, RESULT_DIR
from backtest_portfolio import TW_FEE_BUY, TW_FEE_SELL

SPLIT = "2025-01-01"
QUICK_DAYS = 14


def load_OC():
    files = sorted(f for f in CACHE_DIR.glob("*.parquet") if f.stem.endswith("_TW") or f.stem.endswith("_TWO"))
    o, c = {}, {}
    for f in files:
        try:
            d = pd.read_parquet(f, columns=["Open", "Close"])
            d.index = pd.to_datetime(d.index).tz_localize(None)
            o[f.stem], c[f.stem] = d["Open"], d["Close"]
        except Exception:
            pass
    O = pd.DataFrame(o).sort_index()
    C = pd.DataFrame(c).reindex(O.index)
    return O.where(O > 0), C.where(C > 0)


def add_features(tr: pd.DataFrame, C: pd.DataFrame) -> pd.DataFrame:
    idx = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(C.index)}
    cols = {c: j for j, c in enumerate(C.columns)}
    arr = C.values
    sma20 = C.rolling(20, min_periods=15).mean().values
    feats = {"ret1": [], "ret5": [], "ret20": [], "ext20": []}
    for r in tr.itertuples():
        # 回測 CSV 的 signal_date 多半是「出場」訊號日；進場訊號日 = 進場日的前一個交易日
        i, j = idx.get(r.entry_date), cols.get(r.symbol.replace(".", "_"))
        i = i - 1 if i is not None else None
        if i is None or j is None or i < 21:
            for k in feats:
                feats[k].append(np.nan)
            continue
        c0 = arr[i, j]
        feats["ret1"].append((c0 / arr[i - 1, j] - 1) * 100)
        feats["ret5"].append((c0 / arr[i - 5, j] - 1) * 100)
        feats["ret20"].append((c0 / arr[i - 20, j] - 1) * 100)
        feats["ext20"].append((c0 / sma20[i, j] - 1) * 100)
    for k, v in feats.items():
        tr[k] = v
    return tr


FEATURES = [
    ("rsi", "RSI"), ("vol_ratio", "量比"), ("atr_pct", "波動 ATR%"), ("dist_high", "距52週高(負=低於高點)"),
    ("dist_low", "距52週低"), ("price", "股價"), ("ret1", "訊號日漲幅%"), ("ret5", "近5日漲幅%"),
    ("ret20", "近20日漲幅%"), ("ext20", "乖離20日均線%"),
]


def quintile_table(df, col, cuts):
    q = pd.cut(df[col], cuts, labels=False, include_lowest=True)
    rows = []
    for k in range(5):
        g = df[q == k]
        if len(g) < 30:
            rows.append(None)
            continue
        rows.append(dict(n=len(g), quick=g.quick.mean() * 100, loss=(g.return_pct < 0).mean() * 100,
                         ret=g.return_pct.mean(), exc=g.excess.mean()))
    return q, rows


def main():
    tr = pd.read_csv(sorted(RESULT_DIR.glob("signals_tw_2*.csv"))[-1])
    O, C = load_OC()
    fee_rt = TW_FEE_BUY + TW_FEE_SELL
    tr["base"] = baseline_returns(tr, O, fee_rt)
    tr = add_features(tr, C).dropna(subset=["base", "return_pct"]).copy()
    tr["excess"] = tr["return_pct"] - tr["base"]
    tr["quick"] = ((tr["days_held"] <= QUICK_DAYS) & (tr["return_pct"] < 0)).astype(float)
    tr["month"] = pd.to_datetime(tr["entry_date"]).dt.to_period("M").astype(str)
    train = tr[tr.entry_date < SPLIT].copy()
    test = tr[tr.entry_date >= SPLIT].copy()
    print(f"[資料] 台股回測 {len(tr)} 筆；訓練 {len(train)}／驗證 {len(test)}；"
          f"假訊號(<= {QUICK_DAYS} 天且虧損)佔 {tr.quick.mean()*100:.1f}%（訓練 {train.quick.mean()*100:.1f}%／驗證 {test.quick.mean()*100:.1f}%）")
    print(f"       假訊號平均報酬 {tr[tr.quick==1].return_pct.mean():+.2f}%；其餘 {tr[tr.quick==0].return_pct.mean():+.2f}%\n")

    stable = []
    for col, label in FEATURES:
        cuts = np.unique(np.nanquantile(train[col], [0, .2, .4, .6, .8, 1]))
        if len(cuts) < 6:
            continue
        qtr, rtr = quintile_table(train, col, cuts)
        qte, rte = quintile_table(test, col, cuts)
        print(f"── {label}（切點 {', '.join(f'{c:.1f}' for c in cuts[1:-1])}）")
        print(f"   {'分位':<5}{'訓練 n':>8}{'假訊號%':>8}{'虧損%':>7}{'均報酬':>8}{'超額':>7}   {'驗證 n':>7}{'假訊號%':>8}{'虧損%':>7}{'均報酬':>8}{'超額':>7}")
        for k in range(5):
            a, b = rtr[k], rte[k]
            fa = f"{a['n']:>8}{a['quick']:>8.1f}{a['loss']:>7.1f}{a['ret']:>+8.2f}{a['exc']:>+7.2f}" if a else " " * 38
            fb = f"{b['n']:>10}{b['quick']:>8.1f}{b['loss']:>7.1f}{b['ret']:>+8.2f}{b['exc']:>+7.2f}" if b else ""
            print(f"   Q{k+1:<4}{fa}   {fb}")
        res = []
        for nm, d, q in (("訓練", train, qtr), ("驗證", test, qte)):
            hi, lo = d[q == 4], d[q == 0]
            if len(hi) < 30 or len(lo) < 30:
                res.append(None); continue
            pe, el, eh, pv = boot_diff(hi, lo, "excess")
            pq, ql, qh, qp = boot_diff(hi, lo, "quick")
            res.append((pe, el, eh, pv, pq, ql, qh, qp))
        if all(res):
            (a, b) = res
            print(f"   Q5−Q1 超額差：訓練 {a[0]:+.2f} [{a[1]:+.2f},{a[2]:+.2f}] p={a[3]:.3f}｜驗證 {b[0]:+.2f} [{b[1]:+.2f},{b[2]:+.2f}] p={b[3]:.3f}")
            print(f"   Q5−Q1 假訊號率差(百分點)：訓練 {a[4]*100:+.1f} [{a[5]*100:+.1f},{a[6]*100:+.1f}]｜驗證 {b[4]*100:+.1f} [{b[5]*100:+.1f},{b[6]*100:+.1f}]")
            same = np.sign(a[0]) == np.sign(b[0]) and a[3] < 0.05 and b[3] < 0.10
            same_q = np.sign(a[4]) == np.sign(b[4]) and a[7] < 0.05 and b[7] < 0.10
            if same or same_q:
                stable.append((label, col, same, same_q))
        print()

    print("=== 訓練/驗證一致且顯著的特徵（超額 或 假訊號率）===")
    if not stable:
        print("  （無）")
    for label, col, s1, s2 in stable:
        print(f"  {label}：超額{'✔' if s1 else '✘'}  假訊號率{'✔' if s2 else '✘'}")
    return tr, train, test, stable


if __name__ == "__main__":
    main()
