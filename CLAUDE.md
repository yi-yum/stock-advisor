# Stock Advisor - Claude Code 設定

## 每次開啟對話時，請自動執行以下動作

### 1. 建立台股分析排程（平日 17:00）

使用 CronCreate 工具，參數如下：
- cron: `0 17 * * 1-5`
- recurring: true
- durable: true

prompt 內容：
```
執行台股每日 Claude 分析。

【重要限制】
- 禁止使用 Agent 工具或背景 agents，所有分析直接在本對話完成
- 使用 Bash 工具執行 Python，不用 Write 工具寫檔

步驟 0：刷新今日籌碼資料（確保三大法人資料已更新）
```python
import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, r'C:\Users\User\stock-advisor')
from sector_flow import get_sector_flow
result = get_sector_flow()
print(f"籌碼資料已更新：{result.get('date', '')}，{len(result.get('sectors', []))} 個板塊")
```

步驟 1：用 Bash 執行以下 Python 取得今日清單與所有提示詞：
```python
import sys, json, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, r'C:\Users\User\stock-advisor')
from claude_scanner_analysis import get_scan_candidates, build_buy_prompt, build_watch_prompt
data = get_scan_candidates('tw')
print(f"BUY: {len(data['buy'])}, WATCH: {len(data['watch'])}, date: {data['date_str']}")
for s in data['buy']:
    print(f"===BUY:{s['symbol']}===")
    print(build_buy_prompt(s, 'tw'))
for s in data['watch']:
    print(f"===WATCH:{s['symbol']}===")
    print(build_watch_prompt(s, 'tw'))
```

步驟 2：逐支閱讀提示詞並撰寫分析（不設數量上限，全部完成）：
- BUY 格式：【狀態診斷】【支撐壓力分析】【進場時機評估】🟢/🟡/🔴【操作建議】【持倉管理】
- WATCH 格式：【目前缺口】【翻多可能性評估】⭐高/中/低【翻多所需條件】【若不翻多的風險】【操作建議】【風險評級】

步驟 3：用 Bash 執行 Python 儲存，在 Python 程式碼中直接內嵌 JSON 資料：
```python
import sys, json
sys.path.insert(0, r'C:\Users\User\stock-advisor')
from claude_scanner_analysis import save_claude_analysis, get_scan_candidates
data = get_scan_candidates('tw')
# 將分析結果合併回原始股票資料
analyses = { "symbol1": "分析文字", ... }  # 替換為實際資料
results = []
all_stocks = {s['symbol']: s for s in data['buy'] + data['watch']}
for sym, text in analyses.items():
    if sym in all_stocks:
        s = dict(all_stocks[sym])
        s['claude_analysis'] = text
        results.append(s)
save_claude_analysis('tw', results, data['meta'], data['date_str'])
print(f"儲存完成：{len(results)} 支")
```
```

### 2. 建立美股分析排程（平日 21:20）

使用 CronCreate 工具，參數如下：
- cron: `20 21 * * 1-5`
- recurring: true
- durable: true

prompt 內容同台股，market 全部換成 `'us'`。

### 3. 建立每日總結排程（平日 22:00）

使用 CronCreate 工具，參數如下：
- cron: `0 22 * * 1-5`
- recurring: true
- durable: true

prompt 內容：
```
執行每日台美股盤後總結。

【重要限制】
- 禁止使用 Agent 工具或背景 agents，所有分析直接在本對話完成
- 使用 Bash 工具執行 Python，不用 Write 工具寫檔

步驟 1：用 Bash 取得總結提示詞：
```python
import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, r'C:\Users\User\stock-advisor')
from claude_scanner_analysis import build_daily_summary_prompt
prompt = build_daily_summary_prompt()
if not prompt:
    print("NO_DATA")
else:
    # 寫到暫存檔避免 unicode 問題
    with open(r'C:\Users\User\stock-advisor\_tmp_summary.txt', 'w', encoding='utf-8') as f:
        f.write(prompt)
    print(f"提示詞長度：{len(prompt)} 字元，已寫入 _tmp_summary.txt")
```

步驟 2：讀取提示詞並閱讀後撰寫總結：
用 Read 工具讀取 `C:\Users\User\stock-advisor\_tmp_summary.txt`，
然後根據提示詞內容撰寫完整總結，格式包含：
【今日台股精選】【今日美股精選】【需特別留意】【今日市場觀察】

步驟 3：用 Bash 儲存總結：
```python
import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, r'C:\Users\User\stock-advisor')
from claude_scanner_analysis import save_daily_summary
summary = """（在此貼上完整總結文字）"""
path = save_daily_summary(summary)
print(f"總結已儲存：{path}")
```
```

---

## 完整架構、功能、常見問題

系統架構、前端每頁功能、後端核心檔案、API 路由清單、掃描/回測策略邏輯、常見問題
等完整說明，請見同目錄下的 `使用說明.txt`（單一參考來源，避免與本檔重複維護）。

本檔（CLAUDE.md）只保留 Claude Code 需要「自動執行」的排程設定，其餘描述性內容
一律以 `使用說明.txt` 為準。

（Gemini 相關功能與舊版 Streamlit 入口 `main.py`／`claude_compare.py`／
`claude_analyzer.py` 已於 2026-09-28 移除，目前分析完全由 Claude Code 排程負責。）
