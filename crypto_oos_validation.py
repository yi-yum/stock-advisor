"""
crypto_oos_validation.py — 加密貨幣 per-coin 策略（COIN_BEST）的樣本外驗證
COIN_BEST 的策略與參數是在 2023-2026 全期以『Calmar×勝率』網格搜尋挑出，因此 2023-2026 是樣本內。
A. 往前樣本外：用同一組 COIN_BEST 設定跑 2021-2022（從未用來選參數；2022 為熊市），對照『持有該幣』。
B. 走動式驗證：只用 2023–2024 重新做完整優化（4 種策略×網格，同樣評分），結果拿到 2025–2026 測試，
   並對照 COIN_BEST（已看過測試期，偏樂觀）與持有。
成本：FEE 單邊 0.05%（原設定）與 0.15%（含滑價）兩種。快取放 backtest_cache/crypto_oos，不動原快取。
用法：python crypto_oos_validation.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import backtest_crypto as b

CACHE = Path("backtest_cache/crypto_oos"); CACHE.mkdir(parents=True, exist_ok=True)
END = "2026-09-23"


def get(sym, itv, start, end):
    f = CACHE / f"{sym}_{itv}_{start}_{end}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    df = b.fetch_klines(sym, itv, start, end)
    if len(df):
        df.to_parquet(f)
    return df


def vb_itv():
    return "1h"


def run_cfg(sym, cfg, df, df_d=None):
    st = cfg["strategy"]; bull = None
    if cfg.get("trend_filter") and df_d is not None and len(df_d) >= 200:
        ema = df_d["close"].ewm(span=200, adjust=False).mean()
        bull = (df_d["close"] > ema).reindex(df.index, method="ffill").fillna(False)
    if st == "VB":
        return b.backtest_vb(df, atr_thresh=cfg.get("atr_thresh", b.ATR_THRESH), squeeze_bars=cfg.get("squeeze_bars", b.SQUEEZE_BARS),
                             tp_mult=cfg.get("tp_mult", b.TP_MULT), bull_mask=bull)
    if st == "3ST":
        return b.backtest_triple_st(df, long_only=cfg.get("mode", "long_only") == "long_only", stop_loss_pct=cfg.get("stop_loss"), bull_mask=bull)
    if st == "DC":
        return b.backtest_donchian(df, entry_period=cfg["entry_period"], exit_period=cfg["exit_period"], stop_pct=cfg["stop_pct"])
    return b.backtest_ema_cross(df, fast=cfg["fast"], slow=cfg["slow"], trend=cfg["trend"], stop_pct=cfg["stop_pct"], long_only=cfg.get("long_only", False))


def hold(df):
    c = df["close"].values
    ret = (c[-1] / df["open"].values[0] - 1) * 100
    peak = np.maximum.accumulate(c)
    return ret, ((peak - c) / peak).max() * 100


def data_for(sym, cfg, start, end):
    itv = vb_itv() if cfg["strategy"] == "VB" else "4h"
    # 日線額外往前抓 1 年做 EMA200 暖機
    df = get(sym, itv, start, end)
    dd = get(sym, "1d", (pd.Timestamp(start) - pd.Timedelta(days=400)).strftime("%Y-%m-%d"), end) if cfg.get("trend_filter") else None
    return df, dd


def fmt(r):
    return f"{r['total_return_pct']:>+8.1f}% 勝{r['win_rate']:>4.0f}% {r['trades']:>3}筆 DD{r['max_dd_pct']:>5.1f}%"


def part_a():
    print("=" * 100)
    print("A. 往前樣本外：COIN_BEST（2023–26 優化）套用到 2021-01-01 ~ 2022-12-31（熊市含 2022）")
    print("=" * 100)
    for fee in (0.0005, 0.0015):
        b.FEE = fee
        print(f"\n[單邊成本 {fee*100:.2f}%]")
        print(f"  {'幣':<9}{'策略':<5}{'樣本內 2023–26':<38}{'樣本外 2021–22':<38}{'持有 2021–22'}")
        rs_in, rs_out, rs_h = [], [], []
        for sym, cfg in b.COIN_BEST.items():
            try:
                df_o, dd_o = data_for(sym, cfg, "2021-01-01", "2022-12-31")
                df_i, dd_i = data_for(sym, cfg, "2023-01-01", END)
                if len(df_o) < 500 or len(df_i) < 500:
                    print(f"  {sym}: 資料不足"); continue
                r_in, r_out = run_cfg(sym, cfg, df_i, dd_i), run_cfg(sym, cfg, df_o, dd_o)
                h, hdd = hold(df_o)
            except Exception as e:
                print(f"  {sym}: 失敗 {e}"); continue
            print(f"  {sym:<9}{cfg['strategy']:<5}{fmt(r_in):<38}{fmt(r_out):<38}{h:>+8.1f}% DD{hdd:>5.1f}%")
            rs_in.append(r_in["total_return_pct"]); rs_out.append(r_out["total_return_pct"]); rs_h.append(h)
        print(f"  等權平均：樣本內 {np.mean(rs_in):+.1f}%｜樣本外 {np.mean(rs_out):+.1f}%｜持有 {np.mean(rs_h):+.1f}%")


def part_b():
    print("\n" + "=" * 100)
    print("B. 走動式：只用 2023–2024 做完整優化 → 測試 2025-01-01 ~ 2026-09-23")
    print("=" * 100)
    b.FEE = 0.0005
    rows = []
    for sym, best in b.COIN_BEST.items():
        try:
            v_tr, v_te = get(sym, "1h", "2023-01-01", "2024-12-31"), get(sym, "1h", "2025-01-01", END)
            s_tr, s_te = get(sym, "4h", "2023-01-01", "2024-12-31"), get(sym, "4h", "2025-01-01", END)
            if min(len(v_tr), len(v_te), len(s_tr), len(s_te)) < 500:
                print(f"  {sym}: 資料不足"); continue
            cands = {}
            o = b.optimize_vb(v_tr); cands["VB"] = (o["best_score"], {"strategy": "VB", **o["best_params"]}, o["best_result"])
            o = b.optimize_3st(s_tr); c = o["best_cfg"]
            cands["3ST"] = (o["best_score"], {"strategy": "3ST", "mode": "long_only" if c["long_only"] else "long+short", "stop_loss": c["stop_loss_pct"]}, o["best_result"])
            o = b.optimize_donchian(s_tr); cands["DC"] = (o["best_score"], {"strategy": "DC", **o["best_params"]}, o["best_result"])
            o = b.optimize_ema_cross(s_tr); cands["EMA"] = (o["best_score"], {"strategy": "EMA", **o["best_params"]}, o["best_result"])
            name = max(cands, key=lambda k: cands[k][0]); sc, cfg, r_train = cands[name]
            te = v_te if cfg["strategy"] == "VB" else s_te
            r_test = run_cfg(sym, cfg, te)
            # COIN_BEST 於測試期（已看過）
            dfb, ddb = data_for(sym, best, "2025-01-01", END)
            r_best = run_cfg(sym, best, dfb, ddb)
            h, hdd = hold(s_te)
            rows.append((sym, name, r_train, r_test, best["strategy"], r_best, h, hdd))
            print(f"  {sym:<9}訓練選 {name:<4}訓練 {fmt(r_train)} → 測試 {fmt(r_test)}｜COIN_BEST({best['strategy']}) {fmt(r_best)}｜持有 {h:>+7.1f}% DD{hdd:>5.1f}%")
        except Exception as e:
            print(f"  {sym}: 失敗 {e}")
    if rows:
        f = lambda i, k: np.mean([x[i][k] for x in rows])
        print(f"\n  等權平均：訓練(樣本內) {f(2,'total_return_pct'):+.1f}% ｜ 測試(樣本外) {f(3,'total_return_pct'):+.1f}% ｜ COIN_BEST(測試期) {f(5,'total_return_pct'):+.1f}% ｜ 持有 {np.mean([x[6] for x in rows]):+.1f}%")
        print(f"  勝率：訓練 {f(2,'win_rate'):.0f}% → 測試 {f(3,'win_rate'):.0f}%；最大回撤：訓練 {f(2,'max_dd_pct'):.1f}% → 測試 {f(3,'max_dd_pct'):.1f}%（持有 {np.mean([x[7] for x in rows]):.1f}%）")
        b.FEE = 0.0015
        print("  （成本 0.15% 下之測試期重算略，見 A 部分之成本敏感度）")


if __name__ == "__main__":
    part_a()
    part_b()
