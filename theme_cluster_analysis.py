"""
theme_cluster_analysis.py — 同產業/題材「同時翻多」的訊號，表現是否不同？（台股，僅上市股，有 TWSE 產業別者）
定義：某筆進場的『族群熱度』k = 同產業、其他股票在進場日前 4 日～當日（含）也有進場的檔數。
比較 k=0（單兵）/ 1–2 / 3+（族群共振）的：假訊號率、淨報酬、超額（對全市場、對同產業同期平均），
訓練(進場<2025)與驗證(>=2025)分開看；顯著性以進場月分群 bootstrap。
用法：python theme_cluster_analysis.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_cluster_analysis import cluster_stats, boot_diff, baseline_returns, RESULT_DIR
from backtest_portfolio import TW_FEE_BUY, TW_FEE_SELL
from false_signal_analysis import load_OC, SPLIT, QUICK_DAYS
from sector_flow import _fetch_sector_map

FEE = TW_FEE_BUY + TW_FEE_SELL


def main():
    tr = pd.read_csv(sorted(RESULT_DIR.glob("signals_tw_2*.csv"))[-1])
    smap = _fetch_sector_map()
    tr["code"] = tr.symbol.str.replace(r"\.TWO?$", "", regex=True)
    tr["sector"] = tr.code.map(lambda c: smap.get(c, {}).get("sector"))
    tr = tr[tr.sector.notna()].copy()
    O, _ = load_OC()
    tr["base"] = baseline_returns(tr, O, FEE)

    # 同產業同期平均（持有期間，等權，扣成本）
    idx = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(O.index)}
    arr = O.values
    colsec = defaultdict(list)
    for j, c in enumerate(O.columns):
        code = c.split("_")[0]
        s = smap.get(code, {}).get("sector")
        if s:
            colsec[s].append(j)
    cache = {}
    def sec_base(r):
        key = (r.sector, r.entry_date, r.exit_date)
        if key not in cache:
            ie, ix = idx.get(r.entry_date), idx.get(r.exit_date)
            if ie is None or ix is None or ix <= ie:
                cache[key] = np.nan
            else:
                x = arr[ix, colsec[r.sector]] / arr[ie, colsec[r.sector]] - 1
                x = x[np.isfinite(x)]
                cache[key] = (x.mean() - FEE) * 100 if len(x) >= 3 else np.nan
        return cache[key]
    tr["sbase"] = [sec_base(r) for r in tr.itertuples()]

    # 熱度 k：進場前 4 個交易日～當日，同產業其他股票的進場數
    dates = list(O.index.strftime("%Y-%m-%d"))
    di = {d: i for i, d in enumerate(dates)}
    by = defaultdict(list)  # (sector) -> list of (date_idx, code)
    for r in tr.itertuples():
        if r.entry_date in di:
            by[r.sector].append((di[r.entry_date], r.code))
    ks = []
    for r in tr.itertuples():
        i = di.get(r.entry_date)
        ks.append(np.nan if i is None else sum(1 for j, c in by[r.sector] if i - 4 <= j <= i and c != r.code))
    tr["k"] = ks
    tr = tr.dropna(subset=["k", "base", "return_pct"]).copy()
    tr["ex"] = tr.return_pct - tr.base
    tr["sex"] = tr.return_pct - tr.sbase
    tr["quick"] = ((tr.days_held <= QUICK_DAYS) & (tr.return_pct < 0)).astype(float)
    tr["month"] = pd.to_datetime(tr.entry_date).dt.to_period("M").astype(str)
    tr["grp"] = pd.cut(tr.k, [-1, 0, 2, 1e9], labels=["k=0 單兵", "k=1–2", "k>=3 族群共振"])
    print(f"[資料] 台股上市股 {len(tr)} 筆，產業 {tr.sector.nunique()} 個；k 分布：{tr.grp.value_counts().to_dict()}")

    for label, d in (("全期", tr), ("訓練 2021–2024", tr[tr.entry_date < SPLIT]), ("驗證 2025–2026", tr[tr.entry_date >= SPLIT])):
        print(f"\n=== {label} ===")
        print(f"  {'分組':<12}{'n':>6}{'假訊號%':>8}{'均報酬':>8}{'超額(對大盤)[CI]':>26}{'超額(對同產業)':>16}")
        for g in tr.grp.cat.categories:
            x = d[d.grp == g]
            if len(x) < 30:
                continue
            e = cluster_stats(x.ex.values, x.month)
            s = x.sex.dropna()
            print(f"  {g:<12}{len(x):>6}{x.quick.mean()*100:>8.1f}{x.return_pct.mean():>+7.2f}%  {e['mean']:>+6.2f}[{e['lo']:>+5.2f},{e['hi']:>+5.2f}]{s.mean():>+14.2f}%")
        hi, lo = d[d.grp == "k>=3 族群共振"], d[d.grp == "k=0 單兵"]
        if len(hi) > 30 and len(lo) > 30:
            for col, nm in (("ex", "超額(對大盤)"), ("sex", "超額(對同產業)"), ("quick", "假訊號率")):
                pe, el, eh, pv = boot_diff(hi.dropna(subset=[col]), lo.dropna(subset=[col]), col)
                sc = 100 if col == "quick" else 1
                print(f"  共振 − 單兵 {nm}: {pe*sc:+.2f} [{el*sc:+.2f},{eh*sc:+.2f}] p={pv:.3f}")


if __name__ == "__main__":
    main()
