"""
美股選擇權數據模組
提供選擇權鏈、Put/Call比率、最大痛苦點等分析工具
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def get_options_data(symbol: str) -> dict:
    """
    取得美股選擇權關鍵數據

    Args:
        symbol: 股票代碼（如 'AAPL', 'TSLA'）

    Returns:
        {
            "current_price": 150.0,
            "put_call_ratio": 0.85,
            "total_call_oi": 250000,
            "total_put_oi": 212000,
            "pcr_signal": "偏多",
            "top_call_strikes": [
                {"strike": 155, "oi": 15000, "volume": 3000},
                ...
            ],
            "top_put_strikes": [
                {"strike": 145, "oi": 12000, "volume": 2500},
                ...
            ],
            "max_pain": 150.0,
            "nearest_expiry": "2024-01-19",
            "expiry_dates": [...],
            "error": None
        }
    """
    result = {
        "current_price": None,
        "put_call_ratio": None,
        "total_call_oi": 0,
        "total_put_oi": 0,
        "pcr_signal": None,
        "top_call_strikes": [],
        "top_put_strikes": [],
        "max_pain": None,
        "nearest_expiry": None,
        "expiry_dates": [],
        "error": None
    }

    try:
        # 取得股票基本資訊
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")

        if hist.empty:
            result["error"] = f"無法取得 {symbol} 的股價數據"
            return result

        current_price = hist['Close'].iloc[-1]
        result["current_price"] = float(current_price)

        # 取得選擇權到期日
        expiry_dates = ticker.options
        if not expiry_dates:
            result["error"] = f"{symbol} 無選擇權數據（可能為台股/加密貨幣/無液動性股票）"
            return result

        result["expiry_dates"] = expiry_dates
        nearest_expiry = expiry_dates[0]
        result["nearest_expiry"] = nearest_expiry

        # 取得最近到期日的選擇權鏈
        option_chain = ticker.option_chain(nearest_expiry)
        calls_df = option_chain.calls
        puts_df = option_chain.puts

        # 計算 Call 和 Put 的總 OI
        total_call_oi = calls_df['openInterest'].sum()
        total_put_oi = puts_df['openInterest'].sum()

        result["total_call_oi"] = int(total_call_oi)
        result["total_put_oi"] = int(total_put_oi)

        # 計算 Put/Call Ratio
        if total_call_oi > 0:
            pcr = total_put_oi / total_call_oi
            result["put_call_ratio"] = round(pcr, 2)

            # 判斷訊號
            if pcr < 0.7:
                result["pcr_signal"] = "偏多"
            elif pcr > 1.2:
                result["pcr_signal"] = "偏空"
            else:
                result["pcr_signal"] = "中性"
        else:
            result["put_call_ratio"] = None
            result["pcr_signal"] = "數據不足"

        # 取得前5大 Call OI 行權價
        top_calls = calls_df.nlargest(5, 'openInterest')[['strike', 'openInterest', 'volume']]
        result["top_call_strikes"] = [
            {
                "strike": float(row['strike']),
                "oi": int(row['openInterest']),
                "volume": int(row['volume'])
            }
            for _, row in top_calls.iterrows()
        ]

        # 取得前5大 Put OI 行權價
        top_puts = puts_df.nlargest(5, 'openInterest')[['strike', 'openInterest', 'volume']]
        result["top_put_strikes"] = [
            {
                "strike": float(row['strike']),
                "oi": int(row['openInterest']),
                "volume": int(row['volume'])
            }
            for _, row in top_puts.iterrows()
        ]

        # 計算最大痛苦點（Max Pain）
        max_pain = calculate_max_pain(calls_df, puts_df, current_price)
        result["max_pain"] = max_pain

    except Exception as e:
        result["error"] = f"取得選擇權數據時發生錯誤: {str(e)}"

    return result


def calculate_max_pain(calls_df: pd.DataFrame, puts_df: pd.DataFrame, current_price: float) -> Optional[float]:
    """
    計算最大痛苦點 (Max Pain)

    最大痛苦點是指到期日時股價停在該行權價會造成所有期權買方最大總虧損的行權價

    Args:
        calls_df: Call 選擇權DataFrame
        puts_df: Put 選擇權DataFrame
        current_price: 當前股價

    Returns:
        最大痛苦點的行權價
    """
    try:
        # 合併所有行權價
        strikes = set(calls_df['strike'].tolist() + puts_df['strike'].tolist())

        if not strikes:
            return None

        max_pain_loss = {}

        for strike in strikes:
            # Call 買方虧損
            call_row = calls_df[calls_df['strike'] == strike]
            if not call_row.empty:
                call_oi = call_row['openInterest'].values[0]
                # 若股價 > 行權價，買方虧損 = 0，否則虧損 = (行權價 - 股價) * OI
                call_loss = max(0, strike - current_price) * call_oi if strike < current_price else 0
            else:
                call_loss = 0

            # Put 買方虧損
            put_row = puts_df[puts_df['strike'] == strike]
            if not put_row.empty:
                put_oi = put_row['openInterest'].values[0]
                # 若股價 < 行權價，買方虧損 = 0，否則虧損 = (股價 - 行權價) * OI
                put_loss = max(0, current_price - strike) * put_oi if strike > current_price else 0
            else:
                put_loss = 0

            max_pain_loss[strike] = call_loss + put_loss

        # 找出虧損最小的行權價（即最大痛苦點）
        if max_pain_loss:
            max_pain = min(max_pain_loss, key=max_pain_loss.get)
            return float(max_pain)

        return None

    except Exception:
        return None


def get_iv_percentile(symbol: str) -> dict:
    """
    計算隱含波動率百分位（用近月 ATM option 的 implied volatility）

    Args:
        symbol: 股票代碼

    Returns:
        {
            "iv_current": 0.25,
            "iv_description": "中等波動率",
            "error": None
        }
    """
    result = {
        "iv_current": None,
        "iv_description": None,
        "error": None
    }

    try:
        ticker = yf.Ticker(symbol)
        expiry_dates = ticker.options

        if not expiry_dates:
            result["error"] = f"{symbol} 無選擇權數據"
            return result

        # 使用最近到期日
        option_chain = ticker.option_chain(expiry_dates[0])
        calls_df = option_chain.calls

        # 找 ATM Call（最接近當前股價）
        hist = ticker.history(period="1d")
        current_price = hist['Close'].iloc[-1]

        # 找最接近 ATM 的 Call
        calls_df['distance'] = abs(calls_df['strike'] - current_price)
        atm_call = calls_df.nsmallest(1, 'distance')

        if not atm_call.empty:
            iv = atm_call['impliedVolatility'].values[0]
            result["iv_current"] = round(iv, 4)

            # 簡單分類
            if iv < 0.15:
                result["iv_description"] = "低波動率"
            elif iv < 0.25:
                result["iv_description"] = "中等偏低"
            elif iv < 0.40:
                result["iv_description"] = "中等"
            elif iv < 0.60:
                result["iv_description"] = "中等偏高"
            else:
                result["iv_description"] = "高波動率"
        else:
            result["error"] = "無法找到 ATM 選擇權數據"

    except Exception as e:
        result["error"] = f"計算 IV 時發生錯誤: {str(e)}"

    return result


def render_options_section(symbol: str):
    """
    Streamlit UI 函式 - 顯示選擇權分析區塊

    Args:
        symbol: 股票代碼
    """
    st.subheader(f"📊 {symbol} 選擇權分析")

    # 取得選擇權數據
    options_data = get_options_data(symbol)

    if options_data["error"]:
        st.error(f"⚠️ {options_data['error']}")
        return

    # 第一行：PCR 和訊號
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Put/Call Ratio (PCR)",
            f"{options_data['put_call_ratio']}",
            help="Put 未平倉量 / Call 未平倉量"
        )

    with col2:
        signal = options_data['pcr_signal']
        color = "🟢" if signal == "偏多" else "🔴" if signal == "偏空" else "⚫"
        st.metric(
            "訊號",
            f"{color} {signal}",
            help="PCR < 0.7 = 偏多 | PCR > 1.2 = 偏空 | 中間 = 中性"
        )

    with col3:
        st.metric(
            "最大痛苦點",
            f"${options_data['max_pain']:.2f}" if options_data['max_pain'] else "N/A",
            help="到期日時造成期權買方最大虧損的行權價"
        )

    # 第二行：隱含波動率
    iv_data = get_iv_percentile(symbol)
    if not iv_data["error"]:
        col4, col5 = st.columns(2)
        with col4:
            st.metric("隱含波動率 (IV)", f"{iv_data['iv_current']*100:.2f}%")
        with col5:
            st.metric("波動率等級", iv_data['iv_description'])

    # 第三行：OI 分佈圖
    st.write("**選擇權未平倉量分佈：**")

    fig = render_oi_distribution_chart(
        symbol,
        options_data,
    )
    st.plotly_chart(fig, use_container_width=True)

    # 第四行：Call 和 Put 前5大行權價表
    col_calls, col_puts = st.columns(2)

    with col_calls:
        st.write("**前 5 大 Call 未平倉量：**")
        if options_data['top_call_strikes']:
            calls_display = pd.DataFrame(options_data['top_call_strikes'])
            calls_display.columns = ['行權價', 'OI 數量', '成交量']
            st.dataframe(calls_display, hide_index=True, use_container_width=True)
        else:
            st.info("無數據")

    with col_puts:
        st.write("**前 5 大 Put 未平倉量：**")
        if options_data['top_put_strikes']:
            puts_display = pd.DataFrame(options_data['top_put_strikes'])
            puts_display.columns = ['行權價', 'OI 數量', '成交量']
            st.dataframe(puts_display, hide_index=True, use_container_width=True)
        else:
            st.info("無數據")

    # 第五行：基本資訊
    st.write("**選擇權基本資訊：**")
    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.metric("現價", f"${options_data['current_price']:.2f}")

    with info_col2:
        st.metric(
            "總 Call OI",
            f"{options_data['total_call_oi']:,}",
            help="所有 Call 未平倉量合計"
        )

    with info_col3:
        st.metric(
            "總 Put OI",
            f"{options_data['total_put_oi']:,}",
            help="所有 Put 未平倉量合計"
        )

    # 到期日列表
    st.write("**可用到期日：**")
    col_expiry = st.columns(min(5, len(options_data['expiry_dates'])))
    for idx, expiry in enumerate(options_data['expiry_dates'][:5]):
        with col_expiry[idx]:
            st.caption(expiry)


def render_oi_distribution_chart(
    symbol: str,
    options_data: dict
) -> go.Figure:
    """
    用 Plotly 繪製選擇權 OI 分布圖

    Args:
        symbol: 股票代碼
        options_data: 選擇權數據

    Returns:
        Plotly Figure 物件
    """
    try:
        # 取得詳細的選擇權數據用於繪圖
        ticker = yf.Ticker(symbol)
        nearest_expiry = options_data['nearest_expiry']
        option_chain = ticker.option_chain(nearest_expiry)

        calls_df = option_chain.calls
        puts_df = option_chain.puts

        # 準備數據
        fig = go.Figure()

        # 取得所有行權價範圍（以現價為中心）
        current_price = options_data['current_price']
        all_strikes = sorted(set(calls_df['strike'].tolist() + puts_df['strike'].tolist()))

        # 過濾掉過於遠離現價的行權價（避免圖表過大）
        range_pct = 0.3  # 現價上下 30%
        lower_bound = current_price * (1 - range_pct)
        upper_bound = current_price * (1 + range_pct)

        filtered_strikes = [s for s in all_strikes if lower_bound <= s <= upper_bound]

        # 準備 Call OI 數據
        call_ois = []
        put_ois = []

        for strike in filtered_strikes:
            call_row = calls_df[calls_df['strike'] == strike]
            put_row = puts_df[puts_df['strike'] == strike]

            call_oi = call_row['openInterest'].values[0] if not call_row.empty else 0
            put_oi = put_row['openInterest'].values[0] if not put_row.empty else 0

            call_ois.append(call_oi)
            put_ois.append(put_oi)

        # 繪製 Call OI（綠色）
        fig.add_trace(go.Bar(
            x=filtered_strikes,
            y=call_ois,
            name='Call OI',
            marker=dict(color='rgba(0, 200, 100, 0.7)'),
            hovertemplate='<b>Call</b><br>行權價: $%{x:.2f}<br>OI: %{y:,}<extra></extra>'
        ))

        # 繪製 Put OI（紅色，向下）
        fig.add_trace(go.Bar(
            x=filtered_strikes,
            y=[-oi for oi in put_ois],
            name='Put OI',
            marker=dict(color='rgba(255, 50, 50, 0.7)'),
            hovertemplate='<b>Put</b><br>行權價: $%{x:.2f}<br>OI: %{text:,}<extra></extra>',
            text=put_ois
        ))

        # 標記最大痛苦點
        if options_data['max_pain']:
            max_pain = options_data['max_pain']
            fig.add_vline(
                x=max_pain,
                line=dict(color='orange', width=2, dash='dash'),
                annotation_text=f"最大痛苦點: ${max_pain:.2f}",
                annotation_position="top right"
            )

        # 標記當前股價
        fig.add_vline(
            x=current_price,
            line=dict(color='blue', width=2),
            annotation_text=f"現價: ${current_price:.2f}",
            annotation_position="top left"
        )

        # 更新布局
        fig.update_layout(
            title=f"{symbol} 選擇權未平倉量分布 (到期日: {nearest_expiry})",
            xaxis_title="行權價 ($)",
            yaxis_title="未平倉量 (OI)",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )

        return fig

    except Exception as e:
        # 如果出錯，返回空圖表
        fig = go.Figure()
        fig.add_annotation(text=f"無法繪製圖表: {str(e)}", showarrow=False)
        return fig


if __name__ == "__main__":
    # 測試示例
    test_symbol = "AAPL"

    print(f"\n【{test_symbol} 選擇權數據】")
    print("=" * 60)

    data = get_options_data(test_symbol)

    if data["error"]:
        print(f"錯誤: {data['error']}")
    else:
        print(f"現價: ${data['current_price']:.2f}")
        print(f"Put/Call Ratio: {data['put_call_ratio']}")
        print(f"訊號: {data['pcr_signal']}")
        print(f"Call 總 OI: {data['total_call_oi']:,}")
        print(f"Put 總 OI: {data['total_put_oi']:,}")
        print(f"最大痛苦點: ${data['max_pain']:.2f}")
        print(f"最近到期日: {data['nearest_expiry']}")

        print(f"\n前 5 大 Call OI:")
        for call in data['top_call_strikes']:
            print(f"  ${call['strike']:.2f}: OI={call['oi']:,}, 成交={call['volume']:,}")

        print(f"\n前 5 大 Put OI:")
        for put in data['top_put_strikes']:
            print(f"  ${put['strike']:.2f}: OI={put['oi']:,}, 成交={put['volume']:,}")

    print("\n" + "=" * 60)

    iv_data = get_iv_percentile(test_symbol)
    if iv_data["error"]:
        print(f"IV 錯誤: {iv_data['error']}")
    else:
        print(f"隱含波動率: {iv_data['iv_current']*100:.2f}%")
        print(f"波動率等級: {iv_data['iv_description']}")
