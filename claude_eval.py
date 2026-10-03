"""
claude_eval.py — 檢驗 Claude 分析是否有用（A/B 對照＋機率校準）
資料：scan_results/claude_{tw,us}_*.json（含 ab_group、claude_struct）；價格用 yfinance。
報酬：訊號隔日開盤進場 → 持有 H 個交易日收盤（預設 5、10；不足者略過），扣成本；
      再減去「同市場同日所有 BUY/WATCH 的平均」以去除大盤影響。
輸出：
  1. 對照組 vs 分析組（BUY、WATCH 分開）：超額報酬差與 bootstrap 信賴區間（以訊號日分群）
  2. BUY：p_hold 分位 vs 實際『未被洗出』比例（10日內未收盤跌破 invalid_below），及超額報酬
  3. WATCH：p_flip 分位 vs 實際升級為 BUY 的比例（依後續 scan 檔）
只有 v2 提示詞（含 claude_struct）的資料才會進 2、3。
用法：python claude_eval.py
"""
import glob, json, sys
import numpy as np, pandas as pd, yfinance as yf

HOLD = (5, 10)
FEE = {"tw": 0.00585, "us": 0.001}


def load():
    rows, later = [], {}
    for f in sorted(glob.glob("scan_results/claude_*_2*.json")):
        mk, ds = f.split("claude_")[1][:2], f[-13:-5]
        d = json.load(open(f, encoding="utf-8"))
        for r in d["results"]:
            rows.append(dict(mk=mk, date=f"{ds[:4]}-{ds[4:6]}-{ds[6:]}", sym=r["symbol"], sig=r["signal"],
                             grp=r.get("ab_group"), st=r.get("claude_struct") or {}))
    for f in sorted(glob.glob("scan_results/[tu][ws]_2*.json")):
        mk, ds = f.split("/")[-1].split("_")[0], f[-13:-5]
        later.setdefault(mk, {})[f"{ds[:4]}-{ds[4:6]}-{ds[6:]}"] = {r["symbol"] for r in json.load(open(f, encoding="utf-8"))["results"] if r["signal"] == "BUY"}
    return pd.DataFrame(rows), later


def main():
    df, later = load()
    if df.empty:
        sys.exit("無資料")
    px = {}
    for s in df.sym.unique():
        try:
            px[s] = yf.Ticker(s).history(period="6mo", interval="1d")[["Open", "Close"]]
            px[s].index = px[s].index.strftime("%Y-%m-%d")
        except Exception:
            pass

    def fwd(r, h):
        p = px.get(r.sym)
        if p is None or p.empty:
            return np.nan
        idx = [i for i, d in enumerate(p.index) if d > r.date]
        if not idx or idx[0] + h > len(p) - 1:
            return np.nan
        i = idx[0]
        return (p.Close.iloc[i + h] / p.Open.iloc[i] - 1 - FEE[r.mk]) * 100

    def survived(r, h=10):
        inv = r.st.get("invalid_below")
        p = px.get(r.sym)
        if p is None or not isinstance(inv, (int, float)):
            return np.nan
        idx = [i for i, d in enumerate(p.index) if d > r.date]
        if not idx or idx[0] + h > len(p) - 1:
            return np.nan
        return float((p.Close.iloc[idx[0]:idx[0] + h + 1] >= inv).all())

    for h in HOLD:
        df[f"r{h}"] = df.apply(lambda r: fwd(r, h), axis=1)
        df[f"x{h}"] = df[f"r{h}"] - df.groupby(["mk", "date"])[f"r{h}"].transform("mean")

    rng = np.random.default_rng(0)
    print("== 1. A/B：分析組 − 對照組（超額報酬，%）==")
    for sg in ("BUY", "WATCH"):
        for h in HOLD:
            s = df[(df.sig == sg) & df[f"x{h}"].notna() & df.grp.isin(["claude", "control"])]
            a, c = s[s.grp == "claude"], s[s.grp == "control"]
            if len(a) < 10 or len(c) < 10:
                print(f"  {sg} H={h}: 樣本不足（分析 {len(a)} / 對照 {len(c)}）"); continue
            days = sorted(s.date.unique())
            diffs = []
            for _ in range(2000):
                pick = rng.choice(days, len(days))
                ss = pd.concat([s[s.date == d] for d in pick])
                aa, cc = ss[ss.grp == "claude"], ss[ss.grp == "control"]
                if len(aa) and len(cc):
                    diffs.append(aa[f"x{h}"].mean() - cc[f"x{h}"].mean())
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            print(f"  {sg} H={h}: 分析 n={len(a)} {a[f'x{h}'].mean():+.2f}%  對照 n={len(c)} {c[f'x{h}'].mean():+.2f}%  差 {a[f'x{h}'].mean()-c[f'x{h}'].mean():+.2f} [{lo:+.2f},{hi:+.2f}]（訊號日分群）")

    print("\n== 2. BUY 的 p_hold 校準（10 日內未跌破 invalid_below）==")
    b = df[(df.sig == "BUY") & df.st.map(lambda x: "p_hold" in x)].copy()
    b["p"] = b.st.map(lambda x: x["p_hold"])
    b["ok"] = b.apply(survived, axis=1) if len(b) else pd.Series(dtype=float)
    b = b.dropna(subset=["ok"])
    if len(b) < 20:
        print(f"  有結構化且持有期滿的樣本僅 {len(b)} 筆，尚不足")
    else:
        b["bin"] = pd.qcut(b.p, 3, duplicates="drop")
        print(b.groupby("bin", observed=True).agg(n=("ok", "size"), 預測=("p", "mean"), 實際=("ok", lambda x: x.mean() * 100), x10=("x10", "mean")).round(1))

    print("\n== 3. WATCH 的 p_flip 校準（5 個交易日內升級 BUY）==")
    w = df[(df.sig == "WATCH") & df.st.map(lambda x: "p_flip" in x)].copy()
    w["p"] = w.st.map(lambda x: x["p_flip"])

    def flipped(r):
        ds = sorted(d for d in later.get(r.mk, {}) if d > r.date)[:5]
        return float(any(r.sym in later[r.mk][d] for d in ds)) if len(ds) >= 5 else np.nan
    w["ok"] = w.apply(flipped, axis=1) if len(w) else pd.Series(dtype=float)
    w = w.dropna(subset=["ok"])
    if len(w) < 20:
        print(f"  有結構化且觀察期滿的樣本僅 {len(w)} 筆，尚不足")
    else:
        w["bin"] = pd.qcut(w.p, 3, duplicates="drop")
        print(w.groupby("bin", observed=True).agg(n=("ok", "size"), 預測=("p", "mean"), 實際=("ok", lambda x: x.mean() * 100)).round(1))


if __name__ == "__main__":
    main()
