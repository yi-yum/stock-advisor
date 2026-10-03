"""
exit_stop_experiment.py — 停損/移動停利/部位大小實驗（回測交易的價格路徑重放）
所有觸發一律「收盤價判斷、隔日開盤出場」（與策略機制一致，也比盤中停損保守，不假設停損單必定成交）。
規則在原 3 紅出場之外「多加」一條，較早者先出；停損後不再進場（保守）。
  H%   ：收盤 <= 進場價×(1−H%)
  A×k  ：收盤 <= 進場價 − k×ATR（ATR 取進場訊號日 atr_pct，固定）
  T×k  ：移動停利 收盤 <= 持有期間最高收盤 − k×ATR
  BE   ：收盤曾 >= 進場價+2×ATR 之後，收盤 <= 進場價 出場（保本）
部位：EW 等權；IV 以 1/atr_pct 加權（每筆風險接近）。
指標：交易層（均/中位/勝率/大賠<=-10%比例/最差、基準大贏家(>=20%)報酬保留比例、平均持有日）；
      組合層（每日加權平均、扣成本）CAGR/Sharpe/最大回撤，訓練(進場<2025)/驗證分開。
用法：python exit_stop_experiment.py --market tw|us
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_cluster_analysis import CACHE_DIR, RESULT_DIR
from backtest_portfolio import TW_FEE_BUY, TW_FEE_SELL
from backtest_risk_analysis import port_stats

SPLIT = "2025-01-01"


def load(mk):
    o, c = {}, {}
    for f in sorted(CACHE_DIR.glob("*.parquet")):
        if (mk == "tw") != f.stem.endswith(("_TW", "_TWO")) or f.stem.startswith("_"):
            continue
        try:
            d = pd.read_parquet(f, columns=["Open", "Close"])
            d.index = pd.to_datetime(d.index).tz_localize(None)
            o[f.stem], c[f.stem] = d["Open"], d["Close"]
        except Exception:
            pass
    O = pd.DataFrame(o).sort_index(); O = O.where(O > 0)
    return O, pd.DataFrame(c).reindex(O.index).where(lambda x: x > 0)


def rules():
    r = {"基準(3紅出場)": None}
    for h in (6, 8, 10, 12):
        r[f"H{h}%"] = ("H", h / 100)
    for k in (2, 3, 4):
        r[f"A×{k}"] = ("A", k)
    for k in (3, 4, 6):
        r[f"T×{k}"] = ("T", k)
    r["BE(+2ATR後保本)"] = ("BE", 2)
    r["A×3 + T×4"] = ("AT", (3, 4))
    return r


def trigger(rule, ec, path_c, atr):
    """回傳第一個觸發的 offset(0=進場日收盤)，無則 None。path_c = 進場日..出場前一日的收盤。"""
    kind, p = rule
    if kind == "H":
        hit = path_c <= ec * (1 - p)
    elif kind == "A":
        hit = path_c <= ec - p * atr * ec
    elif kind == "T":
        hit = path_c <= np.maximum.accumulate(path_c) - p * atr * ec
    elif kind == "BE":
        armed = np.maximum.accumulate(path_c) >= ec + p * atr * ec
        hit = armed & (path_c <= ec)
    else:  # AT
        a, t = p
        hit = (path_c <= ec - a * atr * ec) | (path_c <= np.maximum.accumulate(path_c) - t * atr * ec)
    w = np.flatnonzero(hit)
    return int(w[0]) if len(w) else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--market", default="tw", choices=["tw", "us"])
    mk = ap.parse_args().market
    fb, fs = (TW_FEE_BUY, TW_FEE_SELL) if mk == "tw" else (0.0005, 0.0005)
    tr = pd.read_csv(sorted(RESULT_DIR.glob("signals_tw_2*.csv" if mk == "tw" else "signals_us_sp500_2*.csv"))[-1])
    tr = tr[~tr.symbol.str.startswith("_")].copy()
    O, C = load(mk)
    ov, cv = O.values, C.values
    di = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(O.index)}
    cj = {c: j for j, c in enumerate(O.columns)}
    n = len(O)
    with np.errstate(all="ignore"):
        Draw = ov[1:] / ov[:-1] - 1
    D = np.where(np.isfinite(Draw), Draw, 0.0)
    uni = np.where(np.isfinite(np.nanmean(Draw, axis=1)), np.nanmean(Draw, axis=1), 0.0)
    ddates = O.index[1:]

    rs = rules()
    base = []
    for r in tr.itertuples():
        ie, ix, j = di.get(r.entry_date), di.get(r.exit_date), cj.get(r.symbol.replace(".", "_"))
        if ie is None or ix is None or j is None or ix <= ie or not np.isfinite(ov[ie, j]) or not np.isfinite(ov[ix, j]):
            continue
        base.append((r.entry_date, ie, ix, j, float(r.atr_pct) if pd.notna(r.atr_pct) else np.nan))
    print(f"[{mk.upper()}] 可重放交易 {len(base)} 筆")

    res = {}
    for name, rule in rs.items():
        rets, days, wser, ssum = [], [], [], np.zeros(n - 1)
        wsum = np.zeros(n - 1)
        wsum_iv = np.zeros(n - 1); ssum_iv = np.zeros(n - 1)
        entry_dates, bigflag = [], []
        for ed, ie, ix, j, atr in base:
            xx = ix
            if rule is not None and np.isfinite(atr) and atr > 0:
                ec = cv[ie, j] if np.isfinite(cv[ie, j]) else ov[ie, j]
                pc = cv[ie:ix, j]
                pc = np.where(np.isfinite(pc), pc, ec)
                # 進場日收盤也要參照進場開盤價計算 ATR/停損基準：用實際進場價 ep
                ep = ov[ie, j]
                off = trigger(rule, ep, pc, atr)
                if off is not None and ie + off + 1 < ix and np.isfinite(ov[ie + off + 1, j]):
                    xx = ie + off + 1
            ret = (ov[xx, j] * (1 - fs)) / (ov[ie, j] * (1 + fb)) - 1
            rets.append(ret * 100); days.append(xx - ie); entry_dates.append(ed)
            seg = D[ie:xx, j].copy(); seg[0] -= fb; seg[-1] -= fs
            ssum[ie:xx] += seg; wsum[ie:xx] += 1
            w = 1 / atr if np.isfinite(atr) and atr > 0 else 1 / 0.03
            ssum_iv[ie:xx] += w * seg; wsum_iv[ie:xx] += w
        res[name] = dict(ret=np.array(rets), days=np.array(days), ed=np.array(entry_dates),
                         ew=np.where(wsum > 0, ssum / np.maximum(wsum, 1), 0.0),
                         iv=np.where(wsum_iv > 0, ssum_iv / np.maximum(wsum_iv, 1e-9), 0.0))
    b0 = res["基準(3紅出場)"]
    bigmask = b0["ret"] >= 20

    for label, per in (("訓練 2021–2024", "tr"), ("驗證 2025–2026", "te")):
        sel = (b0["ed"] < SPLIT) if per == "tr" else (b0["ed"] >= SPLIT)
        dm = (ddates < pd.Timestamp(SPLIT)) if per == "tr" else (ddates >= pd.Timestamp(SPLIT))
        print(f"\n=== {label}（{sel.sum()} 筆）===")
        print(f"  {'規則':<16}{'均報酬':>7}{'中位':>7}{'勝率':>6}{'大賠%':>6}{'最差':>7}{'大贏家保留':>10}{'持有日':>7}"
              f" | {'EW Sharpe':>9}{'MDD':>7}{'CAGR':>7} | {'IV Sharpe':>9}{'MDD':>7}{'CAGR':>7}")
        for name, x in res.items():
            r = x["ret"][sel]
            keep = x["ret"][sel & bigmask].sum() / b0["ret"][sel & bigmask].sum() * 100
            pe, pi = port_stats(x["ew"][dm]), port_stats(x["iv"][dm])
            print(f"  {name:<16}{r.mean():>+6.2f}%{np.median(r):>+6.2f}%{(r>0).mean()*100:>5.0f}%{(r<=-10).mean()*100:>5.1f}%{r.min():>+6.0f}%{keep:>9.0f}%{x['days'][sel].mean():>7.1f}"
                  f" | {pe['sharpe']:>9.2f}{pe['maxdd']:>6.1f}%{pe['cagr']:>6.1f}% | {pi['sharpe']:>9.2f}{pi['maxdd']:>6.1f}%{pi['cagr']:>6.1f}%")
        pu = port_stats(uni[dm])
        print(f"  [基準]全市場等權 Sharpe {pu['sharpe']:.2f} MDD {pu['maxdd']:.1f}% CAGR {pu['cagr']:.1f}%")


if __name__ == "__main__":
    main()
