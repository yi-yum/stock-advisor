import json
import os
import yfinance as yf
import streamlit as st
from datetime import datetime

from json_utils import atomic_write_json

ALERTS_FILE = "alerts.json"


def load_alerts() -> list:
    """載入所有警示"""
    if not os.path.exists(ALERTS_FILE):
        return []
    with open(ALERTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_alerts(alerts: list):
    """保存警示列表"""
    atomic_write_json(ALERTS_FILE, alerts, ensure_ascii=False, indent=2)


def add_alert(symbol: str, alert_type: str, price: float, note: str = "") -> tuple[bool, str]:
    """
    新增警示

    Args:
        symbol: 股票代號
        alert_type: "above" 或 "below"
        price: 目標價格
        note: 備注

    Returns:
        (成功與否, 訊息)
    """
    if not symbol or symbol.strip() == "":
        return False, "代號不能為空"

    if alert_type not in ["above", "below"]:
        return False, "警示類型必須為『突破上限』或『跌破下限』"

    if price <= 0:
        return False, "目標價必須大於 0"

    alerts = load_alerts()

    # 檢查是否已存在相同警示
    for alert in alerts:
        if alert["symbol"] == symbol and alert["alert_type"] == alert_type and alert["price"] == price:
            return False, f"該警示已存在"

    new_alert = {
        "symbol": symbol.upper().strip(),
        "asset_type": "us" if "." not in symbol.upper() else "tw",
        "alert_type": alert_type,
        "price": float(price),
        "note": note,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "triggered": False,
        "triggered_at": None
    }

    alerts.append(new_alert)
    save_alerts(alerts)

    return True, f"✅ 已新增警示：{symbol.upper()} {alert_type}"


def remove_alert(symbol: str, price: float) -> tuple[bool, str]:
    """
    刪除警示

    Args:
        symbol: 股票代號
        price: 目標價格

    Returns:
        (成功與否, 訊息)
    """
    alerts = load_alerts()
    original_len = len(alerts)

    alerts = [a for a in alerts if not (a["symbol"] == symbol.upper() and a["price"] == price)]

    if len(alerts) == original_len:
        return False, "找不到該警示"

    save_alerts(alerts)
    return True, f"✅ 已刪除警示"


def check_alerts(current_prices: dict) -> list:
    """
    檢查警示是否被觸發

    Args:
        current_prices: {"AAPL": 155.0, "2330.TW": 850.0, ...}

    Returns:
        被觸發的警示列表
    """
    alerts = load_alerts()
    triggered_alerts = []

    for alert in alerts:
        if alert["triggered"]:
            continue

        symbol = alert["symbol"]
        if symbol not in current_prices:
            continue

        current_price = current_prices[symbol]
        target_price = alert["price"]
        alert_type = alert["alert_type"]

        should_trigger = False
        if alert_type == "above" and current_price >= target_price:
            should_trigger = True
        elif alert_type == "below" and current_price <= target_price:
            should_trigger = True

        if should_trigger:
            alert["triggered"] = True
            alert["triggered_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            triggered_alerts.append(alert)

    save_alerts(alerts)
    return triggered_alerts


def get_asset_type(symbol: str) -> str:
    """判斷資產類型"""
    if "." in symbol.upper():
        return "台股"
    else:
        return "美股"


def get_display_alert_type(alert_type: str) -> str:
    """轉換警示類型為顯示文字"""
    if alert_type == "above":
        return "突破上限 ⬆️"
    else:
        return "跌破下限 ⬇️"


def render_alerts_tab():
    """Streamlit UI - 警示管理頁籤"""
    st.title("💰 價格警示系統")

    # 上半部：新增警示
    st.header("📝 新增警示")

    col1, col2, col3 = st.columns(3)

    with col1:
        symbol_input = st.text_input(
            "股票代號",
            placeholder="例：AAPL 或 2330.TW",
            key="alert_symbol_input"
        )

    with col2:
        alert_type_input = st.selectbox(
            "警示類型",
            options=["above", "below"],
            format_func=lambda x: "突破上限 ⬆️" if x == "above" else "跌破下限 ⬇️",
            key="alert_type_input"
        )

    with col3:
        price_input = st.number_input(
            "目標價",
            min_value=0.0,
            step=0.01,
            key="alert_price_input"
        )

    note_input = st.text_input(
        "備注 (選填)",
        placeholder="例：年度目標、技術支撐位...",
        key="alert_note_input"
    )

    if st.button("➕ 新增警示", key="add_alert_btn", use_container_width=True):
        success, message = add_alert(symbol_input, alert_type_input, price_input, note_input)
        if success:
            st.success(message)
            st.rerun()
        else:
            st.error(message)

    st.divider()

    # 下半部：警示清單
    st.header("📋 警示清單")

    alerts = load_alerts()

    if not alerts:
        st.info("📭 還沒有任何警示，新增一個吧！")
    else:
        # 分組：待觸發和已觸發
        pending_alerts = [a for a in alerts if not a["triggered"]]
        triggered_alerts = [a for a in alerts if a["triggered"]]

        # 待觸發警示
        if pending_alerts:
            st.subheader(f"⏳ 待觸發警示 ({len(pending_alerts)} 個)")

            for idx, alert in enumerate(pending_alerts):
                with st.container(border=True):
                    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

                    with col1:
                        st.write(f"**代號：** {alert['symbol']}")
                        st.write(f"**資產：** {get_asset_type(alert['symbol'])}")

                    with col2:
                        st.write(f"**類型：** {get_display_alert_type(alert['alert_type'])}")
                        st.write(f"**目標價：** ${alert['price']:.2f}")

                    with col3:
                        st.write(f"**備注：** {alert.get('note', '-')}")
                        st.write(f"**建立時間：** {alert['created_at']}")

                    with col4:
                        if st.button(
                            "🗑️ 刪除",
                            key=f"delete_alert_{idx}_{alert['symbol']}_{alert['price']}",
                            use_container_width=True
                        ):
                            success, msg = remove_alert(alert["symbol"], alert["price"])
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)

        # 已觸發警示
        if triggered_alerts:
            st.subheader(f"🚨 已觸發警示 ({len(triggered_alerts)} 個)")

            for idx, alert in enumerate(triggered_alerts):
                with st.container(border=True):
                    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])

                    with col1:
                        st.write(f"**代號：** {alert['symbol']}")
                        st.write(f"**資產：** {get_asset_type(alert['symbol'])}")

                    with col2:
                        st.write(f"**類型：** {get_display_alert_type(alert['alert_type'])}")
                        st.write(f"**目標價：** ${alert['price']:.2f}")

                    with col3:
                        st.write(f"**備注：** {alert.get('note', '-')}")
                        st.write(f"**觸發時間：** {alert['triggered_at']}")

                    with col4:
                        if st.button(
                            "🗑️ 刪除",
                            key=f"delete_triggered_{idx}_{alert['symbol']}_{alert['price']}",
                            use_container_width=True
                        ):
                            success, msg = remove_alert(alert["symbol"], alert["price"])
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)

    st.divider()

    # 即時檢查按鈕
    st.header("🔍 即時檢查")

    if st.button("🚀 檢查所有待觸發警示", key="check_alerts_btn", use_container_width=True):
        alerts = load_alerts()
        pending_alerts = [a for a in alerts if not a["triggered"]]

        if not pending_alerts:
            st.info("📭 沒有待觸發的警示")
        else:
            # 收集所有不同的 symbol
            symbols = list(set([a["symbol"] for a in pending_alerts]))

            st.info(f"🔄 正在抓取 {len(symbols)} 支股票的現價...")

            current_prices = {}
            fetch_errors = []

            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    price = ticker.fast_info.get("last_price")
                    if price is not None:
                        current_prices[symbol] = price
                    else:
                        fetch_errors.append(f"{symbol}: 無法取得價格")
                except Exception as e:
                    fetch_errors.append(f"{symbol}: {str(e)}")

            if fetch_errors:
                st.warning("⚠️ 部分股票無法取得現價：\n" + "\n".join(fetch_errors))

            # 檢查警示
            triggered = check_alerts(current_prices)

            if triggered:
                st.success(f"🚨 {len(triggered)} 個警示已觸發！")

                for alert in triggered:
                    st.balloons()
                    with st.container(border=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"### 🎯 {alert['symbol']} 已觸發！")
                            st.write(f"警示類型：{get_display_alert_type(alert['alert_type'])}")
                            st.write(f"目標價：${alert['price']:.2f}")
                            st.write(f"備注：{alert.get('note', '-')}")
                        with col2:
                            current_price = current_prices.get(alert["symbol"], 0)
                            st.write(f"**現價：${current_price:.2f}**")
                            if alert["alert_type"] == "above":
                                distance_pct = ((current_price - alert["price"]) / alert["price"]) * 100
                            else:
                                distance_pct = ((alert["price"] - current_price) / alert["price"]) * 100
                            st.write(f"**超過目標價：{distance_pct:+.2f}%**")

            # 顯示所有股票距目標價的距離
            st.subheader("📊 所有股票與目標價距離")

            distance_data = []
            for alert in pending_alerts:
                symbol = alert["symbol"]
                if symbol in current_prices:
                    current_price = current_prices[symbol]
                    target_price = alert["price"]

                    if alert["alert_type"] == "above":
                        distance_pct = ((current_price - target_price) / target_price) * 100
                    else:
                        distance_pct = ((target_price - current_price) / target_price) * 100

                    distance_data.append({
                        "代號": symbol,
                        "資產": get_asset_type(symbol),
                        "類型": get_display_alert_type(alert["alert_type"]),
                        "現價": f"${current_price:.2f}",
                        "目標價": f"${target_price:.2f}",
                        "距離 %": f"{distance_pct:+.2f}%",
                        "備注": alert.get("note", "-")
                    })

            if distance_data:
                import pandas as pd
                df = pd.DataFrame(distance_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ 無法取得任何股票的現價")


if __name__ == "__main__":
    render_alerts_tab()
