"""
momentum_robustness.py — 美股月度動能（A: 12-1 前10%）的穩健性檢查
1. 逐年報酬 vs 全市場等權 / S&P500；贏的月份比例；月分群(區塊)bootstrap 對全市場的超額 CI
2. 前10% vs 後10%（排序本身有沒有資訊）
3. 排除「貢獻最大的 N 檔」後重跑（N=5,10,20：以持有期間貢獻報酬總和排名）
4. 排除 2025–26 行情後（只看 2021–2024）、以及只看 2021–2023
注意：仍受『現存成分股』前瞻/倖存者偏差影響，這裡只能檢查「行情單一/少數股票」問題。
用法：python momentum_robustness.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from momentum_backtest import load, run, index_ret
from backtest_risk_analysis import port_stats, nw_alpha

FEE = 0.0005


def main():
    O, C, V = load("us")
    rank = V.rolling(60, min_periods=40).mean().rank(axis=1, pct=True).values
    with np.errstate(all="ignore"):
        Draw = O.values[1:] / O.values[:-1] - 1
    D = np.where(np.isfinite(Draw), Draw, 0.0)
    uni = np.where(np.isfinite(np.nanmean(Draw, axis=1)), np.nanmean(Draw, axis=1), 0.0)
    dd = O.index[1:]
    hist = []
    r, turn, st = run(O, C, rank, 12, .10, True, FEE, FEE, hist=hist)
    spx = index_ret("^GSPC", O.index)
    m = dd >= O.index[st]
    df = pd.DataFrame({"mom": r, "uni": uni, "spx": spx if spx is not None else 0.0}, index=dd)[m]

    print("== 1. 逐年（日報酬複利）==")
    yr = (1 + df).groupby(df.index.year).prod() - 1
    print((yr * 100).round(1).rename(columns={"mom": "動能A", "uni": "全市場等權", "spx": "S&P500"}).to_string())
    mm = (1 + df).groupby([df.index.year, df.index.month]).prod() - 1
    print(f"贏全市場的月份比例：{(mm.mom > mm.uni).mean()*100:.0f}%（{len(mm)} 個月）；贏 S&P500：{(mm.mom > mm.spx).mean()*100:.0f}%")
    ex = (mm.mom - mm.uni).values
    rng = np.random.default_rng(0)
    bs = [rng.choice(ex, len(ex)).mean() for _ in range(5000)]
    print(f"月超額均值 {ex.mean()*100:+.2f}%/月，95%CI[{np.percentile(bs,2.5)*100:+.2f}, {np.percentile(bs,97.5)*100:+.2f}]")

    print("\n== 2. 前10% vs 後10%（排序是否有資訊）==")
    rb, _, _ = run(O, C, rank, 12, .10, True, FEE, FEE, bottom=True)
    for nm, x in (("前10%", r), ("後10%", rb), ("全市場等權", uni)):
        ps = port_stats(x[m.values] if hasattr(m, "values") else x[m]); print(f"  {nm:<8}CAGR {ps['cagr']:>6.1f}% Sharpe {ps['sharpe']:.2f} MDD {ps['maxdd']:.1f}%")

    print("\n== 3. 排除貢獻最大的 N 檔後重跑（以持有期間對組合的貢獻報酬排名）==")
    contrib = np.zeros(O.shape[1])
    for e, e2, sel in hist:
        contrib[sel] += D[e:e2][:, sel].sum(axis=0) / len(sel)
    order = np.argsort(-contrib)
    print("  貢獻最大 10 檔：" + ", ".join(O.columns[order[:10]]))
    for N in (5, 10, 20):
        keep = np.ones(O.shape[1], bool); keep[order[:N]] = False
        O2, C2, rk2 = O.loc[:, keep], C.loc[:, keep], rank[:, keep]
        with np.errstate(all="ignore"):
            u2 = np.nanmean(O2.values[1:] / O2.values[:-1] - 1, axis=1)
        u2 = np.where(np.isfinite(u2), u2, 0.0)
        r2, _, _ = run(O2, C2, rk2, 12, .10, True, FEE, FEE)
        ps, pu = port_stats(r2[m.values if hasattr(m, 'values') else m]), port_stats(u2[m.values if hasattr(m, 'values') else m])
        a, _, t = nw_alpha(r2[m], u2[m])
        print(f"  排除前{N:>2}檔：動能 CAGR {ps['cagr']:.1f}% Sharpe {ps['sharpe']:.2f}｜剩餘全市場 CAGR {pu['cagr']:.1f}% Sharpe {pu['sharpe']:.2f}｜α {a*252*100:+.1f}%/年(t={t:+.2f})")

    print("\n== 4. 分期（排除 2025–26 行情）==")
    for lab, hi, lo in (("2021–2024", "2025-01-01", None), ("2021–2023", "2024-01-01", None), ("僅 2022–2024", "2025-01-01", "2022-01-01")):
        mk = m.copy() if not hasattr(m, "values") else m.values.copy()
        mk &= dd < pd.Timestamp(hi)
        if lo: mk &= dd >= pd.Timestamp(lo)
        ps, pu = port_stats(r[mk]), port_stats(uni[mk]); a, _, t = nw_alpha(r[mk], uni[mk])
        sp = port_stats(spx[mk]) if spx is not None else None
        print(f"  {lab:<10}動能 CAGR {ps['cagr']:>5.1f}% Sharpe {ps['sharpe']:.2f} MDD {ps['maxdd']:.1f}%｜全市場 {pu['cagr']:.1f}%/{pu['sharpe']:.2f}"
              + (f"｜S&P500 {sp['cagr']:.1f}%/{sp['sharpe']:.2f}" if sp else "") + f"｜α {a*252*100:+.1f}%(t={t:+.2f})")


if __name__ == "__main__":
    main()
