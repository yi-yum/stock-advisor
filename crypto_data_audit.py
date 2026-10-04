"""
crypto_data_audit.py — 加密貨幣回測資料品質檢查＋逐幣結果（含樣本數警示）
1. 資料品質（每幣、1h/4h、三個時段）：實際根數/預期根數、缺口（相鄰間隔>1 根）數量與最長缺口、零成交量根數、
   單根 |漲跌| >25%（疑似壞資料或極端行情）、重複時間戳。
2. 逐幣結果（COIN_BEST 與 U1~U4）：各時段 報酬/最大回撤/勝率/筆數 vs 持有；筆數 <30 標示『樣本少』。
用法：python crypto_data_audit.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import backtest_crypto as b
from crypto_oos_validation import get, run_cfg, hold, data_for, END
from crypto_unified_params import CANDS, PERIODS

STEP = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4)}
MIN_TRADES = 30


def audit():
    print("=" * 110)
    print("1. 資料品質")
    print("=" * 110)
    print(f"  {'幣':<9}{'時框':<5}{'時段':<9}{'起':<11}{'迄':<11}{'根數/預期':>14}{'缺口':>6}{'最長缺口':>10}{'零量':>6}{'極端根':>7}{'重複':>5}")
    flags = []
    for sym in b.COIN_BEST:
        for itv in ("1h", "4h"):
            for pname, (st, en) in PERIODS.items():
                df = get(sym, itv, st, en)
                if df is None or len(df) == 0:
                    print(f"  {sym:<9}{itv:<5}{pname:<9}（無資料）"); flags.append((sym, itv, pname, "無資料")); continue
                step = STEP[itv]
                exp = int((df.index[-1] - df.index[0]) / step) + 1
                d = df.index.to_series().diff().dropna()
                gaps = d[d > step]
                longest = (gaps.max() / step - 1) if len(gaps) else 0
                zero = int((df["volume"] <= 0).sum())
                ext = int((df["close"].pct_change().abs() > 0.25).sum())
                dup = int(df.index.duplicated().sum())
                start_late = df.index[0] > pd.Timestamp(st) + pd.Timedelta(days=7)
                print(f"  {sym:<9}{itv:<5}{pname:<9}{str(df.index[0].date()):<11}{str(df.index[-1].date()):<11}{len(df):>7}/{exp:<6}{len(gaps):>6}{longest:>9.0f}根{zero:>6}{ext:>7}{dup:>5}"
                      + ("  ⚠起點晚於時段" if start_late else ""))
                if len(gaps) or zero or ext or dup or start_late or len(df) < 0.98 * exp:
                    flags.append((sym, itv, pname, f"缺口{len(gaps)}/零量{zero}/極端{ext}/重複{dup}" + ("/起點晚" if start_late else "")))
    print("\n  需留意的組合：" + ("無" if not flags else ""))
    for f in flags:
        print(f"   - {f[0]} {f[1]} {f[2]}: {f[3]}")


def per_coin():
    print("\n" + "=" * 110)
    print(f"2. 逐幣結果（單邊 0.05%；筆數 <{MIN_TRADES} 標『少』）  格式：報酬%/回撤%/勝率%/筆數；持有：報酬%/回撤%")
    print("=" * 110)
    b.FEE = 0.0005
    strategies = {"COIN_BEST": None, **CANDS}
    for sym in b.COIN_BEST:
        print(f"\n[{sym}]  COIN_BEST={b.COIN_BEST[sym]['strategy']}")
        print(f"  {'策略':<18}" + "".join(f"{p:<34}" for p in PERIODS) )
        hrow = []
        for pname, (st, en) in PERIODS.items():
            df4 = get(sym, "4h", st, en)
            hrow.append(hold(df4) if len(df4) else (np.nan, np.nan))
        print(f"  {'持有':<18}" + "".join(f"{h:>+8.0f}% / DD{d:>5.1f}%{'':<13}" for h, d in hrow))
        for name, cfg in strategies.items():
            c = b.COIN_BEST[sym] if cfg is None else cfg
            cells = []
            for pname, (st, en) in PERIODS.items():
                try:
                    df, dd = data_for(sym, c, st, en)
                    r = run_cfg(sym, c, df, dd)
                    tag = "少" if r["trades"] < MIN_TRADES else " "
                    cells.append(f"{r['total_return_pct']:>+7.0f}/{r['max_dd_pct']:>4.0f}/{r['win_rate']:>3.0f}/{r['trades']:>3}{tag}")
                except Exception as e:
                    cells.append("失敗")
            print(f"  {name:<18}" + "".join(f"{c_:<34}" for c_ in cells))


if __name__ == "__main__":
    audit()
    per_coin()
