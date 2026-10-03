"""
backtest_cluster_analysis.py — 三重 SuperTrend 回測的「分群」顯著性檢定
=========================================================================
問題：回測有數萬筆交易，但同一天（或同一段時間）出現的訊號吃的是同一個大盤，
      彼此不獨立，直接用「筆數」算信賴區間會嚴重高估把握度；另外大盤上漲時
      隨便買一籃子股票也會賺，必須扣掉這部分才看得出選股/訊號本身的邊際。

做法：
  1. 對每筆交易，用「同一段持有期間（進場日開盤 → 出場日開盤）全市場平均報酬」
     （同樣扣來回成本）當基準，算超額報酬 = 交易報酬 − 基準報酬。
  2. 以訊號日 / 週 / 月分群，算 cluster-robust 標準誤與 95% 信賴區間
     （持有期平均約 50 天，相鄰訊號日的持有期高度重疊，月分群最保守）。
  3. 檢定各過濾條件（大盤3綠、旺季 10–4 月）相對於不過濾是否真的有差，
     用「以月為單位重抽樣」的 bootstrap。

用法：
  python backtest_cluster_analysis.py --market tw
  python backtest_cluster_analysis.py --market us          （S&P500 版回測）
資料來源：backtest_results/signals_*.csv（回測交易）、backtest_cache/（價格快取）
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_portfolio import load_taiex_state, TW_FEE_BUY, TW_FEE_SELL

CACHE_DIR  = Path("backtest_cache")
RESULT_DIR = Path("backtest_results")
SEASON_MONTHS = {10, 11, 12, 1, 2, 3, 4}


def t_crit(df: int) -> float:
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, max(df, 1)))
    except Exception:
        return 1.96


def cluster_stats(x: np.ndarray, groups: pd.Series) -> dict:
    """平均、天真標準誤、cluster-robust 標準誤與 95% 區間。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    se_naive = x.std(ddof=1) / np.sqrt(n)
    s = pd.Series(x - m).groupby(groups.values).sum()
    g = len(s)
    se_c = np.sqrt(g / max(g - 1, 1) * float((s ** 2).sum())) / n
    h = t_crit(g - 1) * se_c
    return {"n": n, "clusters": g, "mean": m, "se_naive": se_naive, "se_cluster": se_c,
            "lo": m - h, "hi": m + h, "t": m / se_c if se_c > 0 else np.nan,
            "deff": (se_c / se_naive) ** 2 if se_naive > 0 else np.nan}


def fmt(c: dict) -> str:
    return (f"n={c['n']:>6} 群={c['clusters']:>4}  平均 {c['mean']:+6.2f}%  "
            f"95%CI[{c['lo']:+6.2f}, {c['hi']:+6.2f}]  t={c['t']:+5.2f}  設計效應 {c['deff']:5.1f}x")


def load_universe_open(market: str) -> pd.DataFrame:
    """快取內該市場所有標的的日開盤價矩陣（日期 × 標的）。"""
    files = sorted(CACHE_DIR.glob("*.parquet"))
    tw = [f for f in files if f.stem.endswith("_TW") or f.stem.endswith("_TWO")]
    sel = tw if market == "tw" else [f for f in files if f not in tw]
    cols = {}
    for f in sel:
        try:
            d = pd.read_parquet(f, columns=["Open"])
            d.index = pd.to_datetime(d.index).tz_localize(None)
            cols[f.stem] = d["Open"]
        except Exception:
            pass
    mat = pd.DataFrame(cols).sort_index()
    mat = mat.where(mat > 0)
    print(f"[基準] {market.upper()} 全市場價格矩陣：{mat.shape[1]} 檔 × {mat.shape[0]} 日")
    return mat


def baseline_returns(trades: pd.DataFrame, openm: pd.DataFrame, fee_rt: float) -> np.ndarray:
    """每筆交易同一持有期間的全市場平均報酬（%，已扣來回成本）。"""
    idx = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(openm.index)}
    arr = openm.values
    cache, out = {}, np.full(len(trades), np.nan)
    for k, (e, x) in enumerate(zip(trades["entry_date"], trades["exit_date"])):
        key = (e, x)
        if key not in cache:
            ie, ix = idx.get(e), idx.get(x)
            if ie is None or ix is None or ix <= ie:
                cache[key] = np.nan
            else:
                r = arr[ix] / arr[ie] - 1
                r = r[np.isfinite(r)]
                cache[key] = (r.mean() - fee_rt) * 100 if len(r) else np.nan
        out[k] = cache[key]
    return out


def market_green(dates: pd.Series, market: str, start: str, end: str) -> pd.Series:
    import time
    lookup = {}
    for attempt in range(5):   # Yahoo 偶爾限流，抓不到大盤就稍等重試
        lookup = load_taiex_state(start, end, market)
        if lookup:
            break
        time.sleep(8 * (attempt + 1))
    if not lookup:
        raise SystemExit("無法取得大盤資料（Yahoo 限流），請稍後重試")
    keys = sorted(lookup)

    def get(d):
        if d in lookup:
            return lookup[d]
        prev = [k for k in keys if k <= d]
        return lookup[prev[-1]] if prev else np.nan
    return dates.map(get)


def window_lengths(trades: pd.DataFrame, openm: pd.DataFrame) -> np.ndarray:
    """每筆交易持有的交易日數（進場日到出場日的交易日索引差）。"""
    idx = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(openm.index)}
    out = np.full(len(trades), np.nan)
    for k, (e, x) in enumerate(zip(trades["entry_date"], trades["exit_date"])):
        ie, ix = idx.get(e), idx.get(x)
        if ie is not None and ix is not None and ix > ie:
            out[k] = ix - ie
    return out


def unconditional_by_length(openm: pd.DataFrame, lengths, fee_rt: float) -> dict:
    """持有 L 個交易日、起點不挑時機（所有起點平均）的全市場平均報酬（%，已扣成本）。"""
    arr = openm.values
    res = {}
    for L in sorted({int(l) for l in lengths if np.isfinite(l)}):
        if L >= len(arr):
            continue
        r = arr[L:] / arr[:-L] - 1
        r = np.where(np.isfinite(r), r, np.nan)
        res[L] = (np.nanmean(np.nanmean(r, axis=1)) - fee_rt) * 100
    return res


def boot_diff(a: pd.DataFrame, b: pd.DataFrame, col: str, B: int = 5000, seed: int = 0):
    """以『月』為單位重抽樣，檢定 mean(a[col]) − mean(b[col])。"""
    rng = np.random.default_rng(seed)
    ga = a.groupby("month")[col].agg(["sum", "count"])
    gb = b.groupby("month")[col].agg(["sum", "count"])
    months = sorted(set(ga.index) | set(gb.index))
    sa = ga.reindex(months).fillna(0).values
    sb = gb.reindex(months).fillna(0).values
    diffs = []
    for _ in range(B):
        pick = rng.integers(0, len(months), len(months))
        na, nb = sa[pick, 1].sum(), sb[pick, 1].sum()
        if na == 0 or nb == 0:
            continue
        diffs.append(sa[pick, 0].sum() / na - sb[pick, 0].sum() / nb)
    if not diffs:
        return float("nan"), float("nan"), float("nan"), float("nan")
    diffs = np.sort(diffs)
    point = a[col].mean() - b[col].mean()
    lo, hi = diffs[int(len(diffs) * .025)], diffs[int(len(diffs) * .975)]
    p = 2 * min((np.array(diffs) <= 0).mean(), (np.array(diffs) >= 0).mean())
    return point, lo, hi, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="tw", choices=["tw", "us"])
    args = ap.parse_args()
    mk = args.market

    pat = "signals_tw_2*.csv" if mk == "tw" else "signals_us_sp500_2*.csv"
    f = sorted(RESULT_DIR.glob(pat))[-1]
    tr = pd.read_csv(f)
    print(f"[交易] {f.name}：{len(tr)} 筆，{tr.symbol.nunique()} 檔")

    openm = load_universe_open(mk)
    # 注意：回測 CSV 的 signal_date 在 97% 的交易中是「出場」訊號日（backtest_portfolio.py 在出場分支寫入），
    # 不是進場訊號日。進場訊號為收盤確認、次日開盤成交，故進場訊號日 = 進場日的前一個交易日。
    dl = [d.strftime("%Y-%m-%d") for d in openm.index]
    prev = {dl[i]: dl[i - 1] for i in range(1, len(dl))}
    tr["sig_entry"] = tr["entry_date"].map(prev)
    tr = tr.dropna(subset=["sig_entry"]).copy()
    print(f"[交易] 進場訊號日 {tr.sig_entry.nunique()} 個")
    fee_rt = (TW_FEE_BUY + TW_FEE_SELL) if mk == "tw" else 0.0
    tr["base"] = baseline_returns(tr, openm, fee_rt)
    tr = tr.dropna(subset=["base", "return_pct"]).copy()
    tr["excess"] = tr["return_pct"] - tr["base"]
    tr["month"] = pd.to_datetime(tr["sig_entry"]).dt.to_period("M").astype(str)
    tr["week"]  = pd.to_datetime(tr["sig_entry"]).dt.to_period("W").astype(str)
    tr["green"] = market_green(tr["sig_entry"], mk, "2021-01-01", "2026-09-22")
    tr["season"] = tr["entry_month"].isin(SEASON_MONTHS)
    print(f"[基準] 可計算基準的交易：{len(tr)} 筆；基準平均 {tr.base.mean():+.2f}%（持有期全市場平均）\n")

    print("=== A. 整體：原始報酬 vs 超額報酬（分群方式不同，區間寬度不同）===")
    for label, col in (("原始報酬", "return_pct"), ("超額報酬(扣全市場基準)", "excess")):
        print(f"-- {label}")
        for gname, gcol in (("進場訊號日分群", "sig_entry"), ("週分群", "week"), ("月分群(最保守)", "month")):
            print(f"  {gname:<12}", fmt(cluster_stats(tr[col].values, tr[gcol])))
    print(f"\n  交易的勝率：{(tr.return_pct > 0).mean()*100:.1f}%；基準(全市場買入持有)有 "
          f"{(tr.base > 0).mean()*100:.1f}% 的持有期為正\n")

    print("=== B. 逐年（超額報酬，月分群）===")
    for y, g in tr.groupby("year"):
        print(f"  {y}  ", fmt(cluster_stats(g["excess"].values, g["month"])), f"| 原始 {g.return_pct.mean():+.2f}%  基準 {g.base.mean():+.2f}%")

    print("\n=== C. 過濾條件：原始報酬 vs 超額報酬（月分群）===")
    conds = [("不過濾", tr.index == tr.index),
             ("大盤3綠", tr.green == 3),
             ("大盤<3綠", tr.green < 3),
             ("旺季(10–4月)", tr.season),
             ("淡季(5–9月)", ~tr.season),
             ("大盤3綠+旺季", (tr.green == 3) & tr.season),
             ("大盤3綠+淡季", (tr.green == 3) & ~tr.season)]
    for name, mask in conds:
        g = tr[mask]
        if len(g) < 30:
            continue
        r, e = cluster_stats(g["return_pct"].values, g["month"]), cluster_stats(g["excess"].values, g["month"])
        print(f"  {name:<12} n={len(g):>6}  原始 {r['mean']:+6.2f}% [{r['lo']:+5.2f},{r['hi']:+5.2f}] | "
              f"基準 {g.base.mean():+6.2f}% | 超額 {e['mean']:+6.2f}% [{e['lo']:+5.2f},{e['hi']:+5.2f}] t={e['t']:+5.2f}")

    print("\n=== D. 過濾條件是否真的有差（超額報酬差，以月為單位 bootstrap 95% CI）===")
    for name, a_mask, b_mask in (("大盤3綠 vs 其餘", tr.green == 3, tr.green < 3),
                                 ("旺季 vs 淡季", tr.season, ~tr.season),
                                 ("3綠+旺季 vs 其餘全部", (tr.green == 3) & tr.season, ~((tr.green == 3) & tr.season))):
        a, b = tr[a_mask], tr[b_mask]
        for col, lab in (("return_pct", "原始"), ("excess", "超額")):
            p, lo, hi, pv = boot_diff(a, b, col)
            print(f"  {name:<20} {lab}差 {p:+6.2f}%  95%CI[{lo:+6.2f}, {hi:+6.2f}]  p≈{pv:.3f}")

    print("\n=== E. 訊號集中度 ===")
    per_day = tr.groupby("sig_entry").size()
    print(f"  每個訊號日平均 {per_day.mean():.1f} 筆、最多 {per_day.max()} 筆；"
          f"前 5% 的訊號日佔了 {per_day.sort_values(ascending=False).head(max(1, int(len(per_day)*.05))).sum()/len(tr)*100:.0f}% 的交易")

    print("\n=== F. 報酬拆解：一般持有期平均 + 時機 + 選股（月分群 95%CI）===")
    print("   原始報酬 = 持有同樣天數、起點不挑時機的全市場平均 (A)")
    print("            + 時機：這些訊號的持有期間，全市場實際比 A 好/差多少 (B)")
    print("            + 選股：交易本身比同期全市場多賺多少 (C = 超額報酬)")
    tr["L"] = window_lengths(tr, openm)
    U = unconditional_by_length(openm, tr["L"].dropna().unique(), fee_rt)
    tr["uncond"] = tr["L"].map(U)
    tr["timing"] = tr["base"] - tr["uncond"]
    d = tr.dropna(subset=["uncond"])
    print(f"   平均持有 {d.L.mean():.0f} 個交易日；有效交易 {len(d)} 筆")
    for name, mask in (("不過濾", d.index == d.index), ("大盤3綠", d.green == 3), ("大盤<3綠", d.green < 3),
                       ("旺季", d.season), ("淡季", ~d.season), ("大盤3綠+旺季", (d.green == 3) & d.season)):
        g = d[mask]
        if len(g) < 30:
            continue
        c_t = cluster_stats(g["timing"].values, g["month"])
        c_s = cluster_stats(g["excess"].values, g["month"])
        print(f"  {name:<12} 原始 {g.return_pct.mean():+6.2f}% = 一般 {g.uncond.mean():+5.2f}%"
              f" + 時機 {c_t['mean']:+5.2f}% [{c_t['lo']:+5.2f},{c_t['hi']:+5.2f}]"
              f" + 選股 {c_s['mean']:+5.2f}% [{c_s['lo']:+5.2f},{c_s['hi']:+5.2f}]")


if __name__ == "__main__":
    main()
