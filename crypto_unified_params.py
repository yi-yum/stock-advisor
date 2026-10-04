"""
crypto_unified_params.py — 『所有幣用同一組簡單趨勢參數』vs『逐幣優化(COIN_BEST)』vs 持有
預先設定的 4 組候選（不調參）：
  U1 EMA 9/21、趨勢EMA200、停損7%、多空      U2 同 U1 但只做多
  U3 EMA 12/26、趨勢EMA200、停損7%、只做多    U4 三重ST 只做多、停損8%
時段：2021–22（對 COIN_BEST 是真樣本外）、2023–24、2025–26。全部 4h。
另外做『統一參數的走動式選擇』：只用 2023–24 的 報酬/回撤 在 4 組中選一組，套到 2025–26。
指標（9 檔等權平均）：總報酬、最大回撤、勝過持有的幣數；成本 0.05% 與 0.15%。
用法：python crypto_unified_params.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import backtest_crypto as b
from crypto_oos_validation import get, run_cfg, hold, data_for, END

CANDS = {
    "U1 EMA9/21 多空": {"strategy": "EMA", "fast": 9, "slow": 21, "trend": 200, "stop_pct": 0.07, "long_only": False},
    "U2 EMA9/21 只做多": {"strategy": "EMA", "fast": 9, "slow": 21, "trend": 200, "stop_pct": 0.07, "long_only": True},
    "U3 EMA12/26 只做多": {"strategy": "EMA", "fast": 12, "slow": 26, "trend": 200, "stop_pct": 0.07, "long_only": True},
    "U4 3ST 只做多": {"strategy": "3ST", "mode": "long_only", "stop_loss": 0.08},
}
PERIODS = {"2021–22": ("2021-01-01", "2022-12-31"), "2023–24": ("2023-01-01", "2024-12-31"), "2025–26": ("2025-01-01", END)}


def evaluate(cfg_for, fee):
    b.FEE = fee
    out = {}
    for pname, (st, en) in PERIODS.items():
        rows = []
        for sym in b.COIN_BEST:
            cfg = cfg_for(sym)
            try:
                df, dd = data_for(sym, cfg, st, en)
                if len(df) < 500:
                    continue
                r = run_cfg(sym, cfg, df, dd)
                h, hdd = hold(get(sym, "4h", st, en) if cfg["strategy"] == "VB" else df)
                rows.append((r["total_return_pct"], r["max_dd_pct"], h, hdd, r["win_rate"], r["trades"]))
            except Exception as e:
                print(f"  {sym} {pname}: {e}")
        a = np.array(rows)
        out[pname] = dict(ret=a[:, 0].mean(), dd=a[:, 1].mean(), hold=a[:, 2].mean(), hdd=a[:, 3].mean(),
                          beat=int((a[:, 0] > a[:, 2]).sum()), n=len(a), wr=a[:, 4].mean(), tr=a[:, 5].mean())
    return out


def show(title, res):
    print(f"\n  {title}")
    print(f"    {'時段':<9}{'報酬':>9}{'最大回撤':>9}{'持有報酬':>10}{'持有回撤':>9}{'贏持有':>8}{'勝率':>6}{'筆數/幣':>8}")
    for p, x in res.items():
        print(f"    {p:<9}{x['ret']:>+8.1f}%{x['dd']:>8.1f}%{x['hold']:>+9.1f}%{x['hdd']:>8.1f}%{x['beat']:>5}/{x['n']}{x['wr']:>5.0f}%{x['tr']:>8.0f}")


def main():
    for fee in (0.0005, 0.0015):
        print("=" * 90)
        print(f"單邊成本 {fee*100:.2f}%（4h；9 檔等權平均）")
        print("=" * 90)
        allres = {}
        for name, cfg in CANDS.items():
            allres[name] = evaluate(lambda s, c=cfg: c, fee)
            show(name, allres[name])
        show("逐幣優化 COIN_BEST（2023–26 優化；2021–22 為真樣本外，其餘為樣本內）", evaluate(lambda s: b.COIN_BEST[s], fee))
        if fee == 0.0005:
            sel = max(CANDS, key=lambda k: allres[k]["2023–24"]["ret"] / max(allres[k]["2023–24"]["dd"], 1))
            print(f"\n  走動式選擇（只用 2023–24 的 報酬/回撤）→ 選 {sel} → 2025–26 測試："
                  f"報酬 {allres[sel]['2025–26']['ret']:+.1f}%、回撤 {allres[sel]['2025–26']['dd']:.1f}%、持有 {allres[sel]['2025–26']['hold']:+.1f}%")


if __name__ == "__main__":
    main()
