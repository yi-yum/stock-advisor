"""json_utils.py — atomic JSON 檔案寫入

所有會被 API/排程器/前端同時存取的 JSON 檔案（alerts.json、watchlist.json、
scan_tracker.json、掃描結果...）都應該透過 atomic_write_json() 寫入，避免
寫到一半被中斷（Ctrl+C、當機）或多個寫入者同時搶寫同一個檔案時，
留下殘缺、讀不回來的 JSON。

做法：先把完整內容寫到同目錄下的獨立暫存檔，成功後才用 os.replace()
換掉正式檔案 —— os.replace() 本身是原子操作（Windows/POSIX 皆是），
所以正式檔案永遠只會是「完整的舊版本」或「完整的新版本」。
"""

import json
import os
import tempfile
from pathlib import Path


def atomic_write_json(path, data, **json_kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, **json_kwargs)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise
