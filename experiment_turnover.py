"""
experiment_turnover.py — 台股三重 SuperTrend：降低換手/洗盤成本的規則變體實驗
===========================================================================
backtest_risk_analysis.py 顯示台股每個持倉位置每年換手約 7.6 次、成本約 4.5%/年，
扣成本後組合 Sharpe 0.74、alpha -7.7%。本腳本測試能否用較少的交易換到較好的淨績效。

規則變體（皆沿用「訊號日收盤確認 → 次日開盤成交」、同樣的台股成本）：
  進場確認 k：三條 ST 全翻多後，連續 k 根收盤都維持全多才進場（過濾一日即翻回的假訊號）
  冷卻期  c：出場後 c 根 K 棒內不再進場（避免出場後立刻又被洗進去）
  出場確認 m：全翻空（或 <=thr 條多方）連續 m 根才出場

驗證方式：2021–2024 為訓練期、2025–2026 為驗證期；看變體在兩段是否方向一致，
並以「每日報酬差」的 Newey-West t 值檢定相對基準是否真的有改善。
變體很多，單看哪個最好會高估，請以訓練/驗證一致且顯著的為準。

用法：python experiment_turnover.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from signal_engine import _supertrend
from backtest_portfolio import TW_FEE_BUY, TW_FEE_SELL
from backtest_risk_analysis import nw_alpha, port_stats

CACHE_DIR = Path("backtest_cache")
START, END = "2021-01-01", "2026-09-22"
SPLIT = pd.Timestamp("2025-01-01")
ST_PARAMS = [(11, 2.0), (10, 1.0), (12, 3.0)]
ANN = 252

VARIANTS = {
    "V0 基準":                 dict(ec=1, cd=0,  xc=1, thr=0),
    "冷卻 10 根":              dict(ec=1, cd=10, xc=1, thr=0),
    "冷卻 20 根":              dict(ec=1, cd=20, xc=1, thr=0),
    "冷卻 40 根":              dict(ec=1, cd=40, xc=1, thr=0),
    "進場確認 2 根":           dict(ec=2, cd=0,  xc=1, thr=0),
    "進場確認 3 根":           dict(ec=3, cd=0,  xc=1, thr=0),
    "進場確認 5 根":           dict(ec=5, cd=0,  xc=1, thr=0),
    "出場確認 2 根":           dict(ec=1, cd=0,  xc=2, thr=0),
    "出場確認 3 根":           dict(ec=1, cd=0,  xc=3, thr=0),
    "進場2+出場2":             dict(ec=2, cd=0,  xc=2, thr=0),
    "進場2+冷卻20":            dict(ec=2, cd=20, xc=1, thr=0),
    "進場3+出場2+冷卻20":      dict(ec=3, cd=20, xc=2, thr=0),
    "[對照]較快出場(<=1條多)": dict(ec=1, cd=0,  xc=1, thr=1),
}


def load_data():
    files = sorted(f for f in CACHE_DIR.glob("*.parquet") if f.stem.endswith("_TW") or f.stem.endswith("_TWO"))
    series = {}
    for f in files:
        try:
            d = pd.read_parquet(f, columns=["Open", "High", "Low", "Close"])
            d.index = pd.to_datetime(d.index).tz_localize(None)
            d = d.loc[START:END].dropna()
            d = d[(d["Open"] > 0)]
            if len(d) >= 60:
                series[f.stem] = d
        except Exception:
            pass
    master = sorted(set().union(*[s.index for s in series.values()]))
    midx = pd.DatetimeIndex(master)
    names = list(series)
    O = np.full((len(midx), len(names)), np.nan)
    data = []
    for j, nm in enumerate(names):
        d = series[nm]
        pos = midx.get_indexer(d.index)
        O[pos, j] = d["Open"].values
        dirs = [_supertrend(d["High"].values.astype(float), d["Low"].values.astype(float),
                            d["Close"].values.astype(float), p, m)[0] for p, m in ST_PARAMS]
        green = (np.array(dirs) == 1).sum(axis=0).astype(np.int8)
        data.append((j, pos, d["Open"].values.astype(float), green))
    return midx, names, O, data


def simulate(sym, v):
    j, pos, op, g = sym
    n = len(g)
    out = []
    run3 = -10**9 if g[0] == 3 else 0     # 起始即全多不算「新翻多」
    runx = 0
    in_trade, last_exit, ent = False, -10**9, 0
    for i in range(1, n - 1):
        if g[i] == 3:
            if run3 >= 0:
                run3 += 1
        else:
            run3 = 0
        runx = runx + 1 if g[i] <= v["thr"] else 0
        if not in_trade:
            if run3 == v["ec"] and i - last_exit > v["cd"]:
                in_trade, ent = True, i + 1
        elif runx >= v["xc"]:
            out.append((j, ent, i + 1))
            in_trade, last_exit = False, i
    if in_trade:
        out.append((j, ent, n - 1))
    # 轉成 master 索引與報酬
    res = []
    for jj, a, b in out:
        pa, pb = pos[a], pos[b]
        ret = (op[b] * (1 - TW_FEE_SELL)) / (op[a] * (1 + TW_FEE_BUY)) - 1
        res.append((jj, pa, pb, ret * 100))
    return res


def nw_mean_t(x: np.ndarray, lags: int = 5) -> float:
    """序列平均數的 Newey-West t 值。"""
    n = len(x)
    e = x - x.mean()
    v = e @ e / n
    for l in range(1, lags + 1):
        v += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else float("nan")


def run_variant(data, v):
    t = []
    for sym in data:
        t += simulate(sym, v)
    return t


def portfolio(trades, D, n_days):
    ssum, scnt = np.zeros(n_days - 1), np.zeros(n_days - 1)
    for j, pa, pb, _ in trades:
        if pb <= pa:
            continue
        seg = D[pa:pb, j].copy()
        seg[0] -= TW_FEE_BUY
        seg[-1] -= TW_FEE_SELL
        ssum[pa:pb] += seg
        scnt[pa:pb] += 1
    ser = np.where(scnt > 0, ssum / np.maximum(scnt, 1), 0.0)
    return ser, scnt


def main():
    t0 = time.time()
    midx, names, O, data = load_data()
    print(f"[資料] {len(names)} 檔台股 × {len(midx)} 日；計算 ST 用時 {time.time()-t0:.0f}s")
    with np.errstate(all="ignore"):
        Draw = O[1:] / O[:-1] - 1
    D = np.where(np.isfinite(Draw), Draw, 0.0)
    uni = np.nanmean(Draw, axis=1)
    uni = np.where(np.isfinite(uni), uni, 0.0)
    dates_d = midx[1:]                      # 第 t 個報酬屬於 dates_d[t] 之前的持有
    train = dates_d < SPLIT
    first = None
    results = {}
    for name, v in VARIANTS.items():
        tr = run_variant(data, v)
        ser, cnt = portfolio(tr, D, len(midx))
        results[name] = (tr, ser, cnt)
        print(f"  {name}: {len(tr)} 筆", flush=True)

    base_tr, base_ser, base_cnt = results["V0 基準"]
    print(f"\n[驗證] V0 基準：{len(base_tr)} 筆、平均淨報酬 {np.mean([t[3] for t in base_tr]):+.2f}%"
          f"（對照回測 CSV：37,733 筆、+1.99%）")

    for label, mask in (("訓練期 2021–2024", train), ("驗證期 2025–2026", ~train)):
        years = mask.sum() / ANN
        print(f"\n=== {label}（{mask.sum()} 個交易日）===")
        print(f"  {'變體':<22}{'筆數':>7}{'換手/年':>8}{'均報酬':>8}{'CAGR':>8}{'波動':>7}{'Sharpe':>8}{'最大回撤':>9}"
              f"{'ΔCAGR':>8}{'Δ日報酬t':>9}{'α/年':>8}{'α t':>6}")
        bser = base_ser[mask]
        pu = port_stats(uni[mask])
        print(f"  {'[基準]全市場等權(不扣成本)':<22}{'':>7}{'':>8}{'':>8}{pu['cagr']:>7.1f}%{pu['vol']:>6.1f}%"
              f"{pu['sharpe']:>8.2f}{pu['maxdd']:>8.1f}%")
        for name, (tr, ser, cnt) in results.items():
            sel = [t for t in tr if mask[min(t[1], len(mask) - 1)]]
            active = cnt[mask] > 0
            slots = cnt[mask][active].mean() if active.any() else 1
            turn = len(sel) / slots / years
            ps = port_stats(ser[mask])
            diff = ser[mask] - bser
            if name == "V0 基準":
                dcagr, tt = 0.0, float("nan")
            else:
                dcagr = ps["cagr"] - port_stats(bser)["cagr"]
                tt = nw_mean_t(diff)
            a, _, ta = nw_alpha(ser[mask], uni[mask])
            print(f"  {name:<22}{len(sel):>7}{turn:>8.1f}{np.mean([t[3] for t in sel]):>7.2f}%{ps['cagr']:>7.1f}%"
                  f"{ps['vol']:>6.1f}%{ps['sharpe']:>8.2f}{ps['maxdd']:>8.1f}%{dcagr:>+7.1f}%{tt:>9.2f}"
                  f"{a*ANN*100:>+7.1f}%{ta:>6.2f}")
    print("\n  換手/年 = 平均每個同時持倉位置每年的來回次數；Δ日報酬t = 與 V0 的每日報酬差之 Newey-West t 值；α = 對全市場等權的年化 alpha（Newey-West t）")
    print("  注意：組合為每日再平衡的等權重近似（未模擬資金限制）；變體多，請看訓練/驗證是否一致。")


if __name__ == "__main__":
    main()
