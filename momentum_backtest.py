"""
momentum_backtest.py — 月度橫截面動能（事先寫死的參數，不調參）
規則：每月最後一個交易日收盤算分數 = 過去 L 個月報酬（排除最近 1 個月，即 12-1 或 6-1），
      在『近60日成交值排名前 80%』的股票中取分數最高的前 q%（等權），次一交易日開盤換倉並持有到下個月換倉。
成本：換倉的買入/賣出權重 × 單邊費率（台股買0.1425%、賣0.4425%；美股單邊0.05%）。
基準：全市場等權（日再平衡、不扣成本偏樂觀）、加權指數/S&P500 指數（價格報酬）。
注意：價格快取為現存股票清單 → 有倖存者偏差，績效偏樂觀；日再平衡等權為近似。
預先設定的 4 組：A 12-1 前10%；B 6-1 前10%；C 12-1 前20%；D 同 A 但不做流動性過濾。
用法：python momentum_backtest.py --market tw|us
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))
from backtest_cluster_analysis import CACHE_DIR
from backtest_portfolio import TW_FEE_BUY, TW_FEE_SELL
from backtest_risk_analysis import nw_alpha, port_stats

SPLIT = "2025-01-01"
MDAYS = 21


def load(mk):
    o, c, v = {}, {}, {}
    for f in sorted(CACHE_DIR.glob("*.parquet")):
        if (mk == "tw") != f.stem.endswith(("_TW", "_TWO")) or f.stem.startswith("_"):
            continue
        try:
            d = pd.read_parquet(f, columns=["Open", "Close", "Volume"])
            d.index = pd.to_datetime(d.index).tz_localize(None)
            o[f.stem], c[f.stem], v[f.stem] = d["Open"], d["Close"], d["Close"] * d["Volume"]
        except Exception:
            pass
    O = pd.DataFrame(o).sort_index()
    return O.where(O > 0), pd.DataFrame(c).reindex(O.index).where(lambda x: x > 0), pd.DataFrame(v).reindex(O.index)


def run(O, C, ADV_rank, L_months, top, liq, fb, fs):
    ov, cv = O.values, C.values
    n, m = ov.shape
    with np.errstate(all="ignore"):
        Draw = ov[1:] / ov[:-1] - 1
    D = np.where(np.isfinite(Draw), Draw, 0.0)
    idx = pd.Series(range(n), index=O.index)
    month_end = idx.groupby([O.index.year, O.index.month]).max().values
    L = L_months * MDAYS
    ret = np.zeros(n - 1); held = np.zeros(m, bool); turn = []
    prev_w = np.zeros(m)
    starts = [t for t in month_end if t >= L + 1 and t + 1 < n]
    for k, t in enumerate(starts):
        e = t + 1
        e2 = starts[k + 1] + 1 if k + 1 < len(starts) else n - 1
        with np.errstate(all="ignore"):
            score = cv[t - MDAYS] / cv[t - L] - 1
        ok = np.isfinite(score) & np.isfinite(ov[e])
        if liq:
            ok &= ADV_rank[t] > 0.2
        cand = np.flatnonzero(ok)
        if len(cand) < 30:
            continue
        sel = cand[np.argsort(-score[cand])[: max(10, int(len(cand) * top))]]
        w = np.zeros(m); w[sel] = 1 / len(sel)
        buy = np.clip(w - prev_w, 0, None).sum(); sell = np.clip(prev_w - w, 0, None).sum()
        turn.append(buy)
        seg = D[e:e2][:, sel].mean(axis=1)
        seg[0] -= buy * fb + sell * fs
        ret[e:e2] = seg
        prev_w = w
    return ret, np.array(turn), starts[0] + 1 if starts else n


def index_ret(ticker, dates):
    try:
        s = yf.Ticker(ticker).history(period="max", auto_adjust=True)["Open"]
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return (s / s.shift(1) - 1).reindex(dates[1:]).fillna(0).values
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--market", default="tw", choices=["tw", "us"])
    mk = ap.parse_args().market
    fb, fs = (TW_FEE_BUY, TW_FEE_SELL) if mk == "tw" else (0.0005, 0.0005)
    O, C, V = load(mk)
    rank = V.rolling(60, min_periods=40).mean().rank(axis=1, pct=True).values
    print(f"[{mk.upper()}] {O.shape[1]} 檔 × {O.shape[0]} 日（{O.index[0].date()} ~ {O.index[-1].date()}）")
    with np.errstate(all="ignore"):
        Draw = O.values[1:] / O.values[:-1] - 1
    uni = np.where(np.isfinite(np.nanmean(Draw, axis=1)), np.nanmean(Draw, axis=1), 0.0)
    ddates = O.index[1:]
    cfg = {"A 12-1 前10%": (12, .10, True), "B 6-1 前10%": (6, .10, True), "C 12-1 前20%": (12, .20, True), "D 12-1 前10%(無流動性過濾)": (12, .10, False)}
    out = {k: run(O, C, rank, *v, fb, fs) for k, v in cfg.items()}
    start = max(v[2] for v in out.values())
    bench = {"全市場等權": uni}
    ix = index_ret("^TWII" if mk == "tw" else "^GSPC", O.index)
    if ix is not None:
        bench["加權指數" if mk == "tw" else "S&P500"] = ix
    for label, lo, hi in (("全期", None, None), ("訓練 2021–2024", None, SPLIT), ("驗證 2025–2026", SPLIT, None)):
        m = np.ones(len(ddates), bool)
        m &= ddates >= O.index[start]
        if lo: m &= ddates >= pd.Timestamp(lo)
        if hi: m &= ddates < pd.Timestamp(hi)
        print(f"\n=== {label}（{m.sum()} 日）===")
        print(f"  {'策略':<22}{'CAGR':>7}{'Sharpe':>7}{'MDD':>8}{'α/年(t) 對全市場':>20}{'年換手':>8}")
        for name, (r, turn, _) in out.items():
            ps = port_stats(r[m]); a, _, t = nw_alpha(r[m], uni[m])
            print(f"  {name:<22}{ps['cagr']:>6.1f}%{ps['sharpe']:>7.2f}{ps['maxdd']:>7.1f}%{a*252*100:>+12.1f}%({t:>+5.2f}){turn.mean()*12*100:>7.0f}%")
        for name, r in bench.items():
            ps = port_stats(r[m])
            print(f"  [基準]{name:<16}{ps['cagr']:>6.1f}%{ps['sharpe']:>7.2f}{ps['maxdd']:>7.1f}%")


if __name__ == "__main__":
    main()
