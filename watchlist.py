import json
import os
from datetime import datetime
import streamlit as st
import pandas as pd
from stock_data import get_stock_data, detect_asset_type
from twse_realtime import get_multiple_twse_realtime, is_tw_market_open
from json_utils import atomic_write_json

WATCHLIST_FILE = "watchlist.json"


def load_watchlist() -> list:
    """讀取自選股清單

    回傳格式：
    [
        {
            "symbol": "AAPL",
            "asset_type": "us",
            "note": "長期持有",
            "added_at": "2025-09-13 10:30"
        },
        ...
    ]
    """
    if not os.path.exists(WATCHLIST_FILE):
        # 首次使用時創建預設清單
        return _create_default_watchlist()

    try:
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return _create_default_watchlist()


def _create_default_watchlist() -> list:
    """創建預設自選股清單供測試"""
    default = [
        {
            "symbol": "AAPL",
            "asset_type": "us",
            "note": "科技股龍頭，動能追蹤",
            "added_at": datetime.now().strftime("%Y-%m-%d %H:%M")
        },
        {
            "symbol": "2330.TW",
            "asset_type": "taiwan",
            "note": "台灣半導體龍頭",
            "added_at": datetime.now().strftime("%Y-%m-%d %H:%M")
        },
        {
            "symbol": "BTC",
            "asset_type": "crypto",
            "note": "加密貨幣旗艦幣種",
            "added_at": datetime.now().strftime("%Y-%m-%d %H:%M")
        },
    ]
    save_watchlist(default)
    return default


def save_watchlist(watchlist: list):
    """儲存自選股清單到 JSON 檔案"""
    atomic_write_json(WATCHLIST_FILE, watchlist, ensure_ascii=False, indent=2)


def add_to_watchlist(symbol: str, note: str = "") -> tuple[bool, str]:
    """新增股票到自選股清單

    Args:
        symbol: 股票代號
        note: 備注（加入原因）

    Returns:
        (success: bool, message: str)
    """
    symbol = symbol.upper().strip()

    if not symbol:
        return False, "代號不能為空"

    watchlist = load_watchlist()

    # 檢查是否已存在
    if any(item["symbol"] == symbol for item in watchlist):
        return False, f"{symbol} 已在自選股清單中"

    # 偵測資產類型
    asset_type = detect_asset_type(symbol)

    new_item = {
        "symbol": symbol,
        "asset_type": asset_type,
        "note": note.strip(),
        "added_at": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

    watchlist.append(new_item)
    save_watchlist(watchlist)

    return True, f"✓ 已加入 {symbol}（{asset_type}）"


def remove_from_watchlist(symbol: str) -> tuple[bool, str]:
    """刪除自選股

    Args:
        symbol: 股票代號

    Returns:
        (success: bool, message: str)
    """
    symbol = symbol.upper().strip()
    watchlist = load_watchlist()
    original_len = len(watchlist)

    watchlist = [item for item in watchlist if item["symbol"] != symbol]

    if len(watchlist) == original_len:
        return False, f"在清單中找不到 {symbol}"

    save_watchlist(watchlist)
    return True, f"✓ 已刪除 {symbol}"


def render_watchlist_tab():
    """Streamlit UI 函式 - 自選股管理與掃描"""

    asset_labels = {
        "us": "🇺🇸 美股",
        "taiwan": "🇹🇼 台股",
        "crypto": "🪙 加密貨幣",
    }

    st.markdown("## 📌 自選股清單")

    # ════════════════════════════════════════════
    # 上半部：管理區
    # ════════════════════════════════════════════

    st.markdown("### ➕ 加入自選股")

    col1, col2 = st.columns([2, 1])

    with col1:
        symbols_input = st.text_area(
            "輸入代號（每行一個）",
            placeholder="AAPL\n2330.TW\nBTC",
            height=100,
            label_visibility="collapsed"
        )

    with col2:
        st.markdown("#### 備注")
        note_input = st.text_area(
            "加入原因（選填）",
            placeholder="e.g. 長期持有\n動能追蹤",
            height=100,
            label_visibility="collapsed"
        )

    if st.button("✅ 加入清單", type="primary", use_container_width=True):
        if not symbols_input.strip():
            st.error("請輸入至少一個代號")
        else:
            symbols_list = [s.strip() for s in symbols_input.split("\n") if s.strip()]
            notes_list = [n.strip() for n in note_input.split("\n") if n.strip()]

            added_count = 0
            for i, symbol in enumerate(symbols_list):
                note = notes_list[i] if i < len(notes_list) else ""
                success, message = add_to_watchlist(symbol, note)
                if success:
                    st.success(message)
                    added_count += 1
                else:
                    st.warning(message)

            if added_count > 0:
                st.rerun()

    st.divider()

    # ────────────────────────────────────────────
    # 顯示目前清單
    # ────────────────────────────────────────────

    st.markdown("### 📋 目前清單")

    watchlist = load_watchlist()

    if not watchlist:
        st.info("自選股清單為空，請加入股票")
    else:
        # 按資產類型分組
        asset_groups = {}
        for item in watchlist:
            asset_type = item["asset_type"]
            if asset_type not in asset_groups:
                asset_groups[asset_type] = []
            asset_groups[asset_type].append(item)

        # 顯示各群組
        asset_labels = {
            "us": "🇺🇸 美股",
            "taiwan": "🇹🇼 台股",
            "crypto": "🪙 加密貨幣",
        }

        for asset_type in ["us", "taiwan", "crypto"]:
            if asset_type not in asset_groups:
                continue

            items = asset_groups[asset_type]
            label = asset_labels.get(asset_type, asset_type)

            with st.expander(f"{label} ({len(items)} 支)", expanded=True):
                cols = st.columns([1, 2, 2, 1])
                cols[0].write("**代號**")
                cols[1].write("**備注**")
                cols[2].write("**加入時間**")
                cols[3].write("**操作**")

                st.divider()

                for item in items:
                    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])

                    col1.write(f"`{item['symbol']}`")
                    col2.write(item.get("note", "—"))
                    col3.write(item.get("added_at", "—"))

                    if col4.button("🗑️", key=f"del_{item['symbol']}", help=f"刪除 {item['symbol']}"):
                        success, message = remove_from_watchlist(item["symbol"])
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)

    st.divider()

    # ════════════════════════════════════════════
    # 下半部：快速掃描儀表板
    # ════════════════════════════════════════════

    st.markdown("### 🔍 快速掃描儀表板")

    if not watchlist:
        st.info("自選股清單為空，無法掃描")
        return

    # 台股即時報價按鈕（僅在交易時間顯示）
    col_scan, col_realtime = st.columns([1, 1])

    with col_scan:
        scan_btn = st.button("🚀 掃描自選股", type="primary", use_container_width=True)

    with col_realtime:
        if is_tw_market_open():
            realtime_btn = st.button("🔴 台股即時報價", type="secondary", use_container_width=True)
        else:
            # 非交易時間時顯示灰色按鈕（禁用）
            realtime_btn = False
            st.button("🔴 台股即時報價", type="secondary", use_container_width=True, disabled=True)

    # 台股即時報價功能
    if realtime_btn or "twse_realtime_data" in st.session_state:
        if realtime_btn:  # 按下按鈕時重新取得數據
            # 提取自選股中的台股代號
            tw_stocks = [item["symbol"] for item in watchlist if item["asset_type"] == "taiwan"]

            if not tw_stocks:
                st.warning("自選股清單中沒有台股")
            else:
                with st.spinner("正在取得台股即時報價..."):
                    realtime_data = get_multiple_twse_realtime(tw_stocks)
                    st.session_state.twse_realtime_data = realtime_data

        # 顯示台股即時報價卡片
        if "twse_realtime_data" in st.session_state:
            realtime_data = st.session_state.twse_realtime_data

            st.markdown("#### 📍 台股即時報價")

            # 過濾出有效數據的股票
            valid_data = {k: v for k, v in realtime_data.items() if v.get("price") is not None}

            if valid_data:
                # 按多行顯示指標
                num_stocks = len(valid_data)
                cols_per_row = 4

                row_idx = 0
                for idx, (code, data) in enumerate(valid_data.items()):
                    if idx % cols_per_row == 0:
                        cols = st.columns(cols_per_row)
                        row_idx = 0

                    with cols[row_idx]:
                        # 計算漲跌顏色
                        change_pct = data.get("change_pct", 0)
                        if change_pct is not None and isinstance(change_pct, (int, float)):
                            if change_pct > 0:
                                delta_color = "off"  # Streamlit 會自動標示為綠色
                                delta_str = f"+{change_pct:.2f}%"
                            elif change_pct < 0:
                                delta_color = "inverse"  # Streamlit 會自動標示為紅色
                                delta_str = f"{change_pct:.2f}%"
                            else:
                                delta_color = "off"
                                delta_str = f"{change_pct:.2f}%"
                        else:
                            delta_color = "off"
                            delta_str = "—"

                        st.metric(
                            label=f"{code} {data.get('name', '').strip()}",
                            value=f"NT$ {data.get('price', '—'):.2f}" if data.get('price') else "—",
                            delta=delta_str,
                            delta_color=delta_color
                        )

                    row_idx += 1

            else:
                # 顯示錯誤信息
                error_stocks = [k for k, v in realtime_data.items() if v.get("error")]
                if error_stocks:
                    st.warning(f"無法取得以下代號的報價: {', '.join(error_stocks)}")
                    for code, data in realtime_data.items():
                        if data.get("error"):
                            st.caption(f"{code}: {data['error']}")

            # 清除台股報價數據按鈕
            if st.button("🔄 清除台股報價", use_container_width=False):
                del st.session_state.twse_realtime_data
                st.rerun()

        st.divider()

    if scan_btn or "watchlist_scan_data" in st.session_state:
        with st.spinner("正在掃描自選股..."):
            scan_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, item in enumerate(watchlist):
                status_text.text(f"正在掃描：{item['symbol']}（{idx + 1}/{len(watchlist)}）")

                stock_data, _, error = get_stock_data(item["symbol"])

                if error:
                    scan_results.append({
                        "symbol": item["symbol"],
                        "asset_type": item["asset_type"],
                        "price": "—",
                        "change": "—",
                        "rsi": "—",
                        "macd": "—",
                        "pct_from_high": "—",
                        "bias": "—",
                        "error": error
                    })
                else:
                    # 計算多時框架偏向
                    timeframes = stock_data.get("timeframes", {})
                    weekly_bias = timeframes.get("weekly", {}).get("bias", "—") if timeframes.get("weekly") else "—"
                    daily_bias = timeframes.get("daily", {}).get("bias", "—") if timeframes.get("daily") else "—"

                    scan_results.append({
                        "symbol": item["symbol"],
                        "asset_type": item["asset_type"],
                        "price": stock_data.get("current_price", "—"),
                        "change": stock_data.get("change_pct", "—"),
                        "rsi": stock_data.get("rsi", "—"),
                        "macd": stock_data.get("macd_histogram", "—"),
                        "pct_from_high": stock_data.get("pct_from_52w_high", "—"),
                        "bias_w": weekly_bias,
                        "bias_d": daily_bias,
                        "error": None
                    })

                progress_bar.progress((idx + 1) / len(watchlist))

            progress_bar.empty()
            status_text.empty()

            # 存入 session_state 以便後續使用
            st.session_state.watchlist_scan_data = scan_results

    # 顯示掃描結果
    if "watchlist_scan_data" in st.session_state:
        scan_results = st.session_state.watchlist_scan_data

        st.markdown("#### 📊 掃描結果")

        # 分組顯示
        for asset_type in ["us", "taiwan", "crypto"]:
            items = [r for r in scan_results if r["asset_type"] == asset_type]
            if not items:
                continue

            label = asset_labels.get(asset_type, asset_type)
            st.markdown(f"**{label}**")

            # 建立表格數據
            rows = []
            for r in items:
                if r["error"]:
                    rows.append({
                        "代號": r["symbol"],
                        "狀態": f"❌ 錯誤：{r['error'][:40]}..."
                    })
                else:
                    # RSI 顏色判斷
                    rsi_val = r["rsi"]
                    if isinstance(rsi_val, (int, float)):
                        if rsi_val < 30:
                            rsi_display = f"🟢 {rsi_val}"
                        elif rsi_val > 70:
                            rsi_display = f"🔴 {rsi_val}"
                        else:
                            rsi_display = f"⚪ {rsi_val}"
                    else:
                        rsi_display = "—"

                    # MACD 方向
                    macd_val = r["macd"]
                    if isinstance(macd_val, (int, float)):
                        macd_dir = "📈 金叉" if macd_val > 0 else "📉 死叉"
                    else:
                        macd_dir = "—"

                    rows.append({
                        "代號": r["symbol"],
                        "價格": f"${r['price']}" if r['price'] != "—" else "—",
                        "漲跌%": f"{r['change']}%" if r['change'] != "—" else "—",
                        "RSI": rsi_display,
                        "MACD": macd_dir,
                        "距52W高點%": f"{r['pct_from_high']}%" if r['pct_from_high'] != "—" else "—",
                        "週線": r["bias_w"],
                        "日線": r["bias_d"],
                    })

            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("")

        # 清除掃描數據按鈕
        if st.button("🔄 清除結果", use_container_width=False):
            del st.session_state.watchlist_scan_data
            st.rerun()


if __name__ == "__main__":
    # 測試用
    render_watchlist_tab()
