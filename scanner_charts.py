import os
import matplotlib
matplotlib.use('Agg')  # 必須在 import pyplot 之前
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

CHART_DIR = Path(__file__).parent / "scan_charts"


def generate_chart_image(symbol: str, df: pd.DataFrame) -> str:
    """
    為指定股票生成 K 線圖（含 RSI/MACD/Volume 子圖），存為 PNG，回傳檔案路徑。
    df 需要包含 columns: Open, High, Low, Close, Volume，index 為 DatetimeIndex。
    失敗時回傳空字串。
    """
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = df.tail(120).copy()
        if len(df) < 30:
            return ""

        # RSI(14)
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0).ewm(span=14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(span=14, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line

        # MA
        ma20 = df["Close"].rolling(20).mean()
        ma50 = df["Close"].rolling(50).mean()
        vol_ma20 = df["Volume"].rolling(20).mean()

        ap_list = [
            mpf.make_addplot(rsi, panel=2, color='purple', ylabel='RSI', width=1.0),
            mpf.make_addplot(pd.Series(70, index=df.index), panel=2, color='red', linestyle='--', width=0.7, alpha=0.6),
            mpf.make_addplot(pd.Series(30, index=df.index), panel=2, color='green', linestyle='--', width=0.7, alpha=0.6),
            mpf.make_addplot(macd_line, panel=3, color='blue', ylabel='MACD', width=1.0),
            mpf.make_addplot(signal_line, panel=3, color='orange', width=0.8),
            mpf.make_addplot(macd_hist, panel=3, type='bar', color='gray', alpha=0.5),
            mpf.make_addplot(ma20, panel=0, color='orange', width=1.0),
            mpf.make_addplot(ma50, panel=0, color='cyan', width=1.0),
            mpf.make_addplot(vol_ma20, panel=1, color='yellow', width=0.8),
        ]

        mc = mpf.make_marketcolors(up='lime', down='red', inherit=True)
        style = mpf.make_mpf_style(
            marketcolors=mc,
            base_mpf_style='nightclouds',
            gridstyle=':',
            gridcolor='gray',
            facecolor='#1a1a2e',
            figcolor='#1a1a2e',
            y_on_right=False,
        )

        date_str = datetime.now().strftime("%Y%m%d")
        safe_symbol = symbol.replace(".", "_").replace("/", "_")
        filepath = CHART_DIR / f"{safe_symbol}_{date_str}.png"

        fig, axes = mpf.plot(
            df,
            type='candle',
            style=style,
            addplot=ap_list,
            volume=True,
            panel_ratios=(3, 1, 1, 1),
            figsize=(14, 9),
            title=f"\n{symbol}",
            returnfig=True,
        )
        fig.savefig(str(filepath), dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close(fig)

        return str(filepath)

    except Exception as e:
        print(f"[scanner_charts] 生成 {symbol} 圖表失敗: {e}")
        try:
            plt.close('all')
        except Exception:
            pass
        return ""
