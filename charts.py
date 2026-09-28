"""
Plotly 互動式 K 線圖表模組
用於 Streamlit 股票分析專案
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st


def create_stock_chart(df: pd.DataFrame, symbol: str, asset_type: str = "us") -> go.Figure:
    """
    建立包含以下子圖的互動式 K 線圖：
    - 主圖（佔 60%高度）：K線 + 布林通道 + MA20/MA50/MA200（美股）或 MA20/MA60（台股）
    - 子圖1（佔 15%）：成交量 bar + VolumeMA20 線
    - 子圖2（佔 12%）：RSI(14)，含 70/30 水平線
    - 子圖3（佔 13%）：MACD Histogram（柱狀圖，正負用紅綠區分）+ MACD/Signal 線

    Parameters
    ----------
    df : pd.DataFrame
        包含 OHLCV 和指標的 DataFrame
    symbol : str
        股票代號
    asset_type : str
        資產類型（"us" 或 "taiwan"）

    Returns
    -------
    go.Figure
        Plotly 互動式圖表物件
    """

    # 顏色主題
    BG_COLOR = "#1a1a2e"
    GRID_COLOR = "#2d2d44"
    MA20_COLOR = "#ff9500"  # 橘色
    MA50_COLOR = "#2196f3"  # 藍色
    MA200_COLOR = "#9c27b0"  # 紫色
    MA60_COLOR = "#ffeb3b"  # 黃色
    BULLISH_COLOR = "#26a69a"  # 綠色
    BEARISH_COLOR = "#ef5350"  # 紅色
    BB_COLOR = "rgba(150, 150, 150, 0.2)"  # 灰色半透明

    # 高度比例（單位：相對）
    height_ratios = [60, 15, 12, 13]

    # 建立子圖
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=height_ratios,
        subplot_titles=("K 線圖", "成交量", "RSI", "MACD")
    )

    # ──────────────────────────────────────────
    # 主圖：K 線
    # ──────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="K 線",
            increasing=dict(fillcolor=BULLISH_COLOR, line=dict(color=BULLISH_COLOR)),
            decreasing=dict(fillcolor=BEARISH_COLOR, line=dict(color=BEARISH_COLOR)),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>" +
                         "開：%{open:.2f}<br>" +
                         "高：%{high:.2f}<br>" +
                         "低：%{low:.2f}<br>" +
                         "收：%{close:.2f}<br>" +
                         "量：%{customdata[0]:,.0f}<extra></extra>",
            customdata=df[["Volume"]].values,
        ),
        row=1, col=1
    )

    # 布林通道
    if "BB_upper" in df.columns and "BB_mid" in df.columns and "BB_lower" in df.columns:
        # 上軌
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_upper"],
                name="布林上軌",
                mode="lines",
                line=dict(color=GRID_COLOR, width=1, dash="dash"),
                hovertemplate="上軌：%{y:.2f}<extra></extra>",
            ),
            row=1, col=1
        )

        # 中軌
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_mid"],
                name="布林中軌",
                mode="lines",
                line=dict(color=GRID_COLOR, width=1),
                hovertemplate="中軌：%{y:.2f}<extra></extra>",
            ),
            row=1, col=1
        )

        # 下軌
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_lower"],
                name="布林下軌",
                mode="lines",
                line=dict(color=GRID_COLOR, width=1, dash="dash"),
                fill="tonexty",
                fillcolor=BB_COLOR,
                hovertemplate="下軌：%{y:.2f}<extra></extra>",
            ),
            row=1, col=1
        )

    # 均線
    if "MA20" in df.columns and not df["MA20"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MA20"],
                name="MA20",
                mode="lines",
                line=dict(color=MA20_COLOR, width=1.5),
                hovertemplate="MA20：%{y:.2f}<extra></extra>",
            ),
            row=1, col=1
        )

    if asset_type == "taiwan":
        # 台股顯示 MA60
        if "MA60" in df.columns and not df["MA60"].isna().all():
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["MA60"],
                    name="MA60",
                    mode="lines",
                    line=dict(color=MA60_COLOR, width=1.5),
                    hovertemplate="MA60：%{y:.2f}<extra></extra>",
                ),
                row=1, col=1
            )
    else:
        # 美股顯示 MA50 和 MA200
        if "MA50" in df.columns and not df["MA50"].isna().all():
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["MA50"],
                    name="MA50",
                    mode="lines",
                    line=dict(color=MA50_COLOR, width=1.5),
                    hovertemplate="MA50：%{y:.2f}<extra></extra>",
                ),
                row=1, col=1
            )

        if "MA200" in df.columns and not df["MA200"].isna().all():
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["MA200"],
                    name="MA200",
                    mode="lines",
                    line=dict(color=MA200_COLOR, width=1.5),
                    hovertemplate="MA200：%{y:.2f}<extra></extra>",
                ),
                row=1, col=1
            )

    # ──────────────────────────────────────────
    # 子圖1：成交量 + VolumeMA20
    # ──────────────────────────────────────────
    colors = [BULLISH_COLOR if df["Close"].iloc[i] >= df["Open"].iloc[i] else BEARISH_COLOR
              for i in range(len(df))]

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="成交量",
            marker=dict(color=colors),
            hovertemplate="量：%{y:,.0f}<extra></extra>",
        ),
        row=2, col=1
    )

    if "VolumeMA20" in df.columns and not df["VolumeMA20"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["VolumeMA20"],
                name="20日均量",
                mode="lines",
                line=dict(color=MA20_COLOR, width=2),
                hovertemplate="均量：%{y:,.0f}<extra></extra>",
            ),
            row=2, col=1
        )

    # ──────────────────────────────────────────
    # 子圖2：RSI
    # ──────────────────────────────────────────
    if "RSI" in df.columns and not df["RSI"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI"],
                name="RSI(14)",
                mode="lines",
                line=dict(color="white", width=2),
                hovertemplate="RSI：%{y:.2f}<extra></extra>",
            ),
            row=3, col=1
        )

        # 70 紅線（超買）
        fig.add_hline(y=70, line_dash="dash", line_color=BEARISH_COLOR,
                      annotation_text="70", annotation_position="right",
                      row=3, col=1)

        # 30 綠線（超賣）
        fig.add_hline(y=30, line_dash="dash", line_color=BULLISH_COLOR,
                      annotation_text="30", annotation_position="right",
                      row=3, col=1)

        # 50 中線
        fig.add_hline(y=50, line_dash="dot", line_color=GRID_COLOR,
                      row=3, col=1)

    # ──────────────────────────────────────────
    # 子圖3：MACD
    # ──────────────────────────────────────────
    if "Histogram" in df.columns:
        # MACD Histogram 柱狀圖（正負用紅綠區分）
        hist_colors = [BULLISH_COLOR if h >= 0 else BEARISH_COLOR
                       for h in df["Histogram"]]

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Histogram"],
                name="MACD Histogram",
                marker=dict(color=hist_colors),
                hovertemplate="柱狀：%{y:.4f}<extra></extra>",
            ),
            row=4, col=1
        )

    # MACD 線
    if "MACD" in df.columns and not df["MACD"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD"],
                name="MACD",
                mode="lines",
                line=dict(color=MA20_COLOR, width=2),
                hovertemplate="MACD：%{y:.4f}<extra></extra>",
            ),
            row=4, col=1
        )

    # Signal 線
    if "Signal" in df.columns and not df["Signal"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Signal"],
                name="Signal",
                mode="lines",
                line=dict(color=BEARISH_COLOR, width=2),
                hovertemplate="信號：%{y:.4f}<extra></extra>",
            ),
            row=4, col=1
        )

    # 0 中線
    fig.add_hline(y=0, line_dash="dash", line_color=GRID_COLOR,
                  row=4, col=1)

    # ──────────────────────────────────────────
    # 圖表佈局和樣式
    # ──────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol} 技術分析圖</b>",
            font=dict(size=18, color="white"),
            x=0.5,
            xanchor="center"
        ),
        height=700,
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(family="微軟正黑體, Arial, sans-serif", color="white", size=11),
        margin=dict(l=50, r=50, t=80, b=50),
    )

    # 更新 x 軸和 y 軸
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=GRID_COLOR,
        zeroline=False,
        row=4, col=1
    )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=GRID_COLOR,
        zeroline=False,
        row=1, col=1
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=GRID_COLOR,
        zeroline=False,
    )

    # 隱藏 rangeslider
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    # 最新指標註解（右上角）
    if len(df) > 0:
        latest = df.iloc[-1]
        latest_date = df.index[-1]

        # 構建註解文本
        annotation_text = f"<b>最新指標 ({latest_date.strftime('%Y-%m-%d')})</b><br>"
        annotation_text += f"收：{latest.get('Close', 'N/A'):.2f}<br>"

        if "RSI" in df.columns and not pd.isna(latest.get("RSI")):
            annotation_text += f"RSI：{latest['RSI']:.2f}<br>"

        if "MACD" in df.columns and not pd.isna(latest.get("MACD")):
            annotation_text += f"MACD：{latest['MACD']:.4f}<br>"

        if "Signal" in df.columns and not pd.isna(latest.get("Signal")):
            annotation_text += f"信號：{latest['Signal']:.4f}<br>"

        if "Histogram" in df.columns and not pd.isna(latest.get("Histogram")):
            annotation_text += f"柱狀：{latest['Histogram']:.4f}"

        fig.add_annotation(
            text=annotation_text,
            xref="paper", yref="paper",
            x=0.98, y=0.98,
            showarrow=False,
            bgcolor="rgba(26, 26, 46, 0.8)",
            bordercolor="white",
            borderwidth=1,
            font=dict(size=10, color="white"),
            align="left",
            xanchor="right",
            yanchor="top",
        )

    return fig


def render_chart_section(df: pd.DataFrame, symbol: str, asset_type: str):
    """
    在 Streamlit 中顯示圖表，包含時間範圍選擇器。

    Parameters
    ----------
    df : pd.DataFrame
        包含 OHLCV 和指標的 DataFrame
    symbol : str
        股票代號
    asset_type : str
        資產類型（"us" 或 "taiwan"）
    """

    st.subheader(f"📊 {symbol} 技術分析圖表")

    # 檢查數據充足性
    if len(df) < 20:
        st.warning(f"⚠️ 數據不足（只有 {len(df)} 根 K 線）。需要至少 20 根 K 線才能渲染圖表。")
        return

    # 時間範圍選擇器
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        range_1m = st.radio(
            "選擇時間範圍",
            ["1 個月", "3 個月", "6 個月", "1 年"],
            index=1,  # 預設選擇 3 個月
            horizontal=False
        )

    # 根據選擇過濾數據
    range_map = {
        "1 個月": 20,
        "3 個月": 60,
        "6 個月": 120,
        "1 年": 252
    }

    lookback_days = range_map[range_1m]
    df_filtered = df.tail(lookback_days).copy()

    # 建立並顯示圖表
    fig = create_stock_chart(df_filtered, symbol, asset_type)

    st.plotly_chart(fig, use_container_width=True)

    # 顯示最新指標摘要
    st.divider()
    latest = df.iloc[-1]

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "收盤價",
            f"{latest.get('Close', 'N/A'):.2f}",
            delta=f"{((latest.get('Close', 0) - df.iloc[-2].get('Close', 1)) / df.iloc[-2].get('Close', 1) * 100):.2f}%" if len(df) >= 2 else None
        )

    with col2:
        if "RSI" in df.columns and not pd.isna(latest.get("RSI")):
            st.metric("RSI(14)", f"{latest['RSI']:.2f}")
        else:
            st.metric("RSI(14)", "N/A")

    with col3:
        if "MACD" in df.columns and not pd.isna(latest.get("MACD")):
            st.metric("MACD", f"{latest['MACD']:.4f}")
        else:
            st.metric("MACD", "N/A")

    with col4:
        if "Signal" in df.columns and not pd.isna(latest.get("Signal")):
            st.metric("Signal", f"{latest['Signal']:.4f}")
        else:
            st.metric("Signal", "N/A")

    with col5:
        if "Histogram" in df.columns and not pd.isna(latest.get("Histogram")):
            st.metric("Histogram", f"{latest['Histogram']:.4f}")
        else:
            st.metric("Histogram", "N/A")
