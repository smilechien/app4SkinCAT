from __future__ import annotations

import hashlib
import html
import io
import json
import math
import os
import random
import re
import zipfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, abort, redirect, render_template_string, request, send_file, send_from_directory, session, url_for

APP_TITLE = "Rasch CAT App of Dermatology Physician Specialist Entry Examination in Taiwan"
DEFAULT_BUNDLE = Path(__file__).with_name("replay_bundle.zip")
SECRET_KEY = os.environ.get("CAT_SECRET_KEY", "rasch-cat-demo-secret")
IMG_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")
OPTION_LABELS = ["A", "B", "C", "D", "E", "F", "G"]

try:
    from gtts import gTTS
except Exception:
    gTTS = None

TTS_CACHE_DIR = Path(tempfile.gettempdir()) / "raschcatskin_tts_cache"
TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

README_FALLBACK = """# Rasch CAT App\n\nThis app reads `response_category.csv` with columns `key, no, link, item, item2`.\n\n- `item` = Chinese item text\n- `item2` = English item text\n- `link` = optional figure / picture / graph path or URL\n- If `link` is a local PNG or image file name, place the file under `pic/`\n- You can store images either inside `replay_bundle.zip` under `pic/` or beside `app.py` in a local `pic/` folder\n\nModes:\n1. CAT mode selects the next item by maximum Rasch information\n2. non-CAT mode presents all items in fixed item-number order like a traditional test\n3. Voice practice mode randomly samples items near a chosen theta range and uses server-generated audio for mobile devices and browser TTS as desktop fallback\n\nThe examinee view does not show the hidden answer key or per-item score in CAT and non-CAT modes.\n"""

LABELS = {
    "en": {
        "home_title": APP_TITLE,
        "home_intro": "This app uses replay_bundle.zip as the item bank.",
        "choose_language": "Language",
        "lang_zh": "Chinese",
        "lang_en": "English",
        "choose_mode": "Test mode",
        "mode_cat": "CAT",
        "mode_linear": "non-CAT",
        "mode_voice": "Voice practice",
        "mode_demo": "Quick 20-item CAT demo",
        "mode_compare": "CAT vs non-CAT",
        "mode_compare_n": "CAT vs non-CAT(n)",
        "start_test": "Start test",
        "start_cat": "Start CAT",
        "start_linear": "Start non-CAT",
        "start_voice": "Start voice practice",
        "start_demo": "Start quick 20-item CAT demo",
        "start_compare": "Start CAT vs non-CAT",
        "start_compare_n": "Start CAT vs non-CAT(n)",
        "sim_cat_number": "Simulated CAT number (comparison)",
        "max_items": "Maximum CAT items",
        "stop_se": "Stop CAT when SE ≤",
        "start_theta": "Starting theta",
        "theta_range": "Theta range +/- (voice practice only)",
        "voice_range_unit": "Range in a unit",
        "start_item": "Starting item (non-CAT)",
        "readme": "README",
        "works": "How the test works",
        "sel_info_cat": "Selected by maximum information from the remaining items.",
        "sel_info_linear": "Presented sequentially from the selected starting item in fixed item-number order.",
        "figure_ref": "Figure / graph reference",
        "open_ref": "Open reference link",
        "submit": "Submit answer",
        "to_home": "To Homepage",
        "responses": "Response history",
        "new_test": "Start over",
        "final_theta": "Final theta",
        "post_se": "Posterior SE",
        "percentile": "Percentile",
        "items_used": "Items used",
        "reason": "Reason",
        "answer_hidden": "The answer key and per-item score are hidden from the examinee view.",
        "item": "Item",
        "difficulty": "Difficulty",
        "your_answer": "Your answer",
        "answer_match": "TRUE/FALSE",
        "theta_after": "Theta after item",
        "se_after": "SE after item",
        "reference": "Reference",
        "open": "Open",
        "question": "Question",
        "home_loaded": "Loaded bundle",
        "home_summary_items": "Items",
        "home_summary_mean": "Person prior mean",
        "home_summary_sd": "Prior SD",
        "structure": "Expected response_category.csv structure",
        "columns": "Columns",
        "kidmap": "KIDMAP dashboard",
        "person_fit": "Person fit",
        "infit_mnsq": "INFIT MNSQ",
        "outfit_mnsq": "OUTFIT MNSQ",
        "item_fit": "Bank item fit",
        "answered_items": "Answered items",
        "mode": "Mode",
        "mode_cat_short": "CAT",
        "mode_linear_short": "non-CAT",
        "mode_voice_short": "Voice",
        "mode_demo_short": "Demo",
        "mode_compare_short": "CAT vs non-CAT",
        "mode_compare_n_short": "CAT vs non-CAT(n)",
        "mode_compare_n_short": "CAT vs non-CAT(n)",
        "voice_intro": "Voice practice randomly samples items within the selected theta range. On mobile, the app first plays server-generated MP3 audio; on desktop, browser TTS is kept as fallback. The stem and options are read first, followed by a pause and then the correct answer.",
        "voice_start_audio": "Start audio cycle",
        "voice_replay": "Replay current item",
        "voice_pause": "Pause audio",
        "voice_resume": "Resume audio",
        "voice_stop": "Stop audio",
        "voice_prev": "Previous item",
        "voice_next": "Next item",
        "voice_auto_next": "Auto-next after answer",
        "voice_pause_seconds": "Pause before answer (sec)",
        "voice_correct": "Correct answer",
        "voice_done": "Voice practice completed.",
        "voice_restart": "Restart sequence",
        "voice_status_ready": "Ready to read the selected items.",
        "voice_actual_items": "Actual items in theta range",
        "voice_requested_items": "Requested maximum items",
        "voice_actual_note": "If the theta range contains fewer items than requested, only the available items in that theta range are used.",
        "trend_chart": "CAT result trend chart",
        "zstd": "ZSTD",
        "itemfit_note": "Gray dots show bank item INFIT ZSTD, while red dots show response-level ZSTD for the answered items. These are related but not identical quantities.",
        "kidmap_note": "KIDMAP residual bubbles are recomputed against the final theta estimate, whereas the green trend line stores stepwise ZSTD at each CAT step. They are related but not numerically identical.",
        "demo_note": "Quick demo CAT uses 20 adaptively selected items. For each item, the app randomly chooses an answer and then shows the CAT result page immediately.",
        "home_note": "In non-CAT mode, items are shown sequentially from the chosen starting item; after the last item, the sequence wraps to Item 1 so every item can be reviewed. Voice practice mode randomly samples items near the selected theta range and reads them aloud with browser TTS. Quick demo CAT runs a fixed 20-item CAT with randomized answers and jumps straight to the result page. Every figure linked in response_category.csv can be checked on the test screen if the file exists.",
        "wright_map_home": "Original simulation Wright Map",
        "wright_map_home_note": "The homepage Wright Map uses bank item INFIT MNSQ on the x-axis, item measure on the y-axis, bubble size proportional to item SE, and a red dotted boundary at INFIT MNSQ = 1.5. Hover each bubble to inspect the item statistics.",
        "go_wright_map": "Go to Wright Map",
    },
    "zh": {
        "home_title": APP_TITLE,
        "home_intro": "本系統使用 replay_bundle.zip 作為題庫。",
        "choose_language": "作答語言",
        "lang_zh": "中文",
        "lang_en": "英文",
        "choose_mode": "測驗模式",
        "mode_cat": "CAT 自適應測驗",
        "mode_linear": "non-CAT 傳統逐題測驗",
        "mode_voice": "語音練習模式",
        "mode_demo": "快速 20 題 CAT 示範",
        "mode_compare": "CAT vs non-CAT 比較",
        "mode_compare_n": "CAT vs non-CAT(n) 比較",
        "start_test": "開始測驗",
        "start_cat": "開始 CAT",
        "start_linear": "開始 non-CAT",
        "start_voice": "開始語音練習",
        "start_demo": "開始快速 20 題 CAT 示範",
        "start_compare": "開始 CAT vs non-CAT 比較",
        "start_compare_n": "開始 CAT vs non-CAT(n) 比較",
        "sim_cat_number": "模擬 CAT 次數（比較用）",
        "max_items": "CAT 最多題數",
        "stop_se": "CAT 當 SE ≤ 時停止",
        "start_theta": "起始能力值 theta",
        "theta_range": "theta 範圍 +/-（僅語音練習）",
        "voice_range_unit": "範圍單位",
        "start_item": "起始題號（僅 non-CAT）",
        "readme": "說明",
        "works": "測驗流程",
        "sel_info_cat": "下一題依目前 theta 的最大資訊量選取。",
        "sel_info_linear": "從指定起始題號開始，依固定題號順序逐題呈現。",
        "figure_ref": "圖表／圖片參考",
        "open_ref": "開啟參考連結",
        "submit": "送出答案",
        "to_home": "回首頁",
        "responses": "作答紀錄",
        "new_test": "重新測驗",
        "final_theta": "最終 theta",
        "post_se": "後驗 SE",
        "percentile": "百分等級",
        "items_used": "已作答題數",
        "reason": "停止原因",
        "answer_hidden": "考生頁面不顯示標準答案與逐題得分。",
        "item": "題目",
        "difficulty": "難度",
        "your_answer": "你的答案",
        "answer_match": "TRUE/FALSE",
        "theta_after": "作答後 theta",
        "se_after": "作答後 SE",
        "reference": "參考圖",
        "open": "開啟",
        "question": "題目",
        "home_loaded": "已載入題庫",
        "home_summary_items": "題數",
        "home_summary_mean": "受試者先驗平均",
        "home_summary_sd": "先驗標準差",
        "structure": "response_category.csv 欄位結構",
        "columns": "欄位",
        "kidmap": "KIDMAP 儀表板",
        "person_fit": "個人配適統計",
        "infit_mnsq": "INFIT MNSQ",
        "outfit_mnsq": "OUTFIT MNSQ",
        "item_fit": "題庫試題配適",
        "answered_items": "已作答題目",
        "mode": "模式",
        "mode_cat_short": "CAT",
        "mode_linear_short": "non-CAT",
        "mode_voice_short": "語音",
        "mode_demo_short": "示範",
        "mode_compare_short": "CAT vs non-CAT",
        "mode_compare_n_short": "CAT vs non-CAT(n)",
        "voice_intro": "語音練習會依所選 theta 範圍隨機抽題。手機優先播放伺服器產生的 MP3 音檔，桌機則保留瀏覽器 TTS 作為備援；系統會先朗讀題幹與選項，停頓後再朗讀正確答案。",
        "voice_start_audio": "開始語音循環",
        "voice_replay": "重播本題",
        "voice_pause": "暫停語音",
        "voice_resume": "繼續語音",
        "voice_stop": "停止語音",
        "voice_prev": "上一題",
        "voice_next": "下一題",
        "voice_auto_next": "答案後自動下一題",
        "voice_pause_seconds": "答案前停頓（秒）",
        "voice_correct": "正確答案",
        "voice_done": "語音練習已完成。",
        "voice_restart": "重新播放序列",
        "voice_status_ready": "已準備好朗讀所選題目。",
        "trend_chart": "CAT 結果折線圖",
        "zstd": "ZSTD",
        "itemfit_note": "灰點顯示題庫原始的 item INFIT ZSTD；紅點改為作答題目的 response-level ZSTD。兩者相關，但不是同一個量。",
        "kidmap_note": "KIDMAP 的殘差泡泡會以最終 theta 重新計算；綠線則保留每一步 CAT 當下的逐步 ZSTD，因此兩者相關但不會完全相同。",
        "demo_note": "快速示範 CAT 會固定執行 20 題自適應測驗；系統每題隨機作答，完成後直接顯示 CAT 結果頁。",
        "home_note": "在 non-CAT 模式下，題目會從指定起始題號開始依固定順序呈現；到最後一題後會回到第 1 題，直到所有題目都完成。語音練習模式會依所選 theta 範圍隨機抽題並以瀏覽器 TTS 朗讀。快速 20 題 CAT 示範會隨機作答並直接顯示結果頁。只要 response_category.csv 的 link 能對到實際檔案，每題的圖片都會在作答頁顯示。",
        "wright_map_home": "原始模擬資料 Wright Map",
        "wright_map_home_note": "首頁 Wright Map 以題庫 item 的 INFIT MNSQ 為 x 軸、item measure 為 y 軸、泡泡大小正比於 item SE，並在 INFIT MNSQ = 1.5 畫出紅色虛線邊界。將滑鼠移到泡泡上可查看題目統計值。",
        "go_wright_map": "前往 Wright Map",
    },
}

HOME_TMPL = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }}</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 1100px; margin: 24px auto; padding: 0 16px; line-height: 1.55; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 18px; margin-bottom: 18px; }
    .muted { color: #555; }
    .btn { display: inline-block; background: #2563eb; color: white; padding: 10px 16px; border-radius: 8px; text-decoration: none; border: 0; cursor: pointer; }
    .btn-secondary { background: #475569; }
    code { background: #f4f4f4; padding: 2px 6px; border-radius: 6px; }
    .inline-fields label { display: inline-block; margin-right: 16px; margin-bottom: 10px; }
    pre { white-space: pre-wrap; word-break: break-word; background: #fafafa; padding: 12px; border-radius: 10px; border: 1px solid #eee; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }
    .modebox { display: inline-block; margin-right: 18px; margin-bottom: 10px; }
    .small { font-size: 0.94rem; }
  </style>
</head>
<body>
  <h1>{{ labels.home_title }}</h1>
  <div class="card">
    <p>{{ labels.home_intro }}</p>
    <ul>
      <li><code>response_category.csv</code> → <code>key, no, link, item, item2</code></li>
      <li><code>fixed_item_delta.csv</code> → Rasch item difficulty</li>
      <li><code>person_estimates.csv</code> → prior mean and SD</li>
      <li><code>pic/</code> → optional local figures for the <code>link</code> column</li>
    </ul>
    <p class="muted">{{ labels.home_loaded }}: <strong>{{ bundle_name }}</strong></p>
    <p class="muted">{{ labels.home_summary_items }}: <strong>{{ summary.n_items }}</strong> | {{ labels.home_summary_mean }}: <strong>{{ '%.3f'|format(summary.prior_mean) }}</strong> | {{ labels.home_summary_sd }}: <strong>{{ '%.3f'|format(summary.prior_sd) }}</strong></p>
    <form method="post" action="{{ url_for('start_test') }}">
      <div class="inline-fields">
        <label>{{ labels.choose_language }}
          <select name="language">
            <option value="zh">{{ labels.lang_zh }}</option>
            <option value="en">{{ labels.lang_en }}</option>
          </select>
        </label>
      </div>
      <div class="inline-fields">
        <label>{{ labels.max_items }} <input type="number" name="max_items" min="5" max="60" value="20"></label>
        <label>{{ labels.stop_se }} <input type="number" step="0.01" name="stop_se" min="0.15" max="1.00" value="0.32"></label>
        <label>{{ labels.start_theta }} <input type="number" step="0.01" name="start_theta" value="{{ '%.2f'|format(summary.prior_mean) }}"></label>
        <label>{{ labels.theta_range }}
          <select name="theta_range">
            <option value="1" selected>1</option>
            <option value="1.5">1.5</option>
            <option value="2">2</option>
            <option value="2.5">2.5</option>
            <option value="3">3</option>
          </select>
        </label>
        <label>{{ labels.start_item }} <input type="number" name="start_item" min="1" max="{{ summary.n_items }}" value="1"></label>
        <label>{{ labels.sim_cat_number }} <input type="number" name="sim_cat_number" min="2" max="{{ summary.n_sim_max }}" value="20"></label>
      </div>
      <div style="margin-top:14px; display:flex; gap:12px; flex-wrap:wrap; align-items:center;">
        <button class="btn" type="submit" name="mode" value="cat">{{ labels.start_cat }}</button>
        <button class="btn" type="submit" name="mode" value="linear">{{ labels.start_linear }}</button>
        <button class="btn" type="submit" name="mode" value="voice">{{ labels.start_voice }}</button>
        <button class="btn" type="submit" name="mode" value="demo">{{ labels.start_demo }}</button>
        <button class="btn" type="submit" name="mode" value="compare">{{ labels.start_compare }}</button>
        <button class="btn" type="submit" name="mode" value="compare_n">{{ labels.start_compare_n }}</button>
        <a class="btn btn-secondary" href="#wright-map-home">{{ labels.go_wright_map }}</a>
      </div>
      <p class="muted small" style="margin-top:10px;">{{ labels.choose_mode }}: <strong>{{ labels.mode_cat }}</strong> / <strong>{{ labels.mode_linear }}</strong> / <strong>{{ labels.mode_voice }}</strong> / <strong>{{ labels.mode_demo }}</strong> / <strong>{{ labels.mode_compare }}</strong> / <strong>{{ labels.mode_compare_n }}</strong></p>
      <p class="muted small">{{ labels.demo_note }}</p>
    </form>
    <p class="muted small" style="margin-top:14px;">{{ labels.home_note }}</p>
  </div>

  <div class="card">
    <h2>{{ labels.structure }}</h2>
    <table>
      <thead><tr><th>{{ labels.columns }}</th><th>Description</th></tr></thead>
      <tbody>
        <tr><td><code>key</code></td><td>Hidden correct answer key, such as A/B/C/D.</td></tr>
        <tr><td><code>no</code></td><td>Item number.</td></tr>
        <tr><td><code>link</code></td><td>Optional figure path or URL. Local image files can be placed in <code>pic/</code>.</td></tr>
        <tr><td><code>item</code></td><td>Chinese item text including options.</td></tr>
        <tr><td><code>item2</code></td><td>English item text including options.</td></tr>
        <tr><td><code>Voice practice</code></td><td>Randomly samples items within the selected theta range and reads the item, options, and correct answer by browser TTS.</td></tr>
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>{{ labels.works }}</h2>
    <ol>
      <li>Choose Chinese or English before starting.</li>
      <li>Choose CAT, non-CAT, voice practice, or quick demo CAT mode on the homepage.</li>
      <li>CAT mode selects the remaining item with maximum Rasch information at the current theta.</li>
      <li>non-CAT mode starts from the selected item number, then continues in ascending item-number order and wraps to Item 1 after the last item.</li>
      <li>Voice practice mode randomly samples items within the chosen theta range and uses browser TTS to read the item, options, pause, and then read the correct answer.</li>
      <li>Quick demo CAT runs a fixed 20-item CAT, randomly answers each selected item, and jumps straight to the result page.</li>
      <li>Ability is updated by EAP on a fixed grid with a normal prior for CAT and non-CAT modes.</li>
      <li>The result page shows final theta, SE, percentile, person INFIT/OUTFIT MNSQ, and the KIDMAP dashboard.</li>
    </ol>
    <p class="muted">{{ labels.answer_hidden }}</p>
  </div>

  <div class="card" id="wright-map-home">
    <h2>{{ labels.wright_map_home }}</h2>
    <div>{{ home_wright_svg|safe }}</div>
    <p class="muted small">{{ labels.wright_map_home_note }}</p>
  </div>

  <div class="card">
    <h2>{{ labels.readme }}</h2>
    <pre>{{ readme_text }}</pre>
  </div>
</body>
</html>
"""

ITEM_TMPL = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }}</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 1100px; margin: 24px auto; padding: 0 16px; line-height: 1.6; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 18px; margin-bottom: 18px; }
    .muted { color: #555; }
    .btn { display: inline-block; background: #2563eb; color: white; padding: 10px 16px; border-radius: 8px; text-decoration: none; border: 0; cursor: pointer; }
    .btn-secondary { background: #475569; }
    .stat { display: inline-block; margin-right: 16px; margin-bottom: 10px; padding: 8px 12px; background: #f7f7f7; border-radius: 10px; }
    .option { display: block; padding: 12px 14px; margin: 12px 0; border: 1px solid #ddd; border-radius: 10px; cursor: pointer; }
    .opt-row { display: flex; align-items: flex-start; gap: 10px; }
    .opt-radio { margin-top: 4px; }
    .opt-label { min-width: 1.8em; font-weight: 700; }
    .opt-text { flex: 1; white-space: pre-wrap; word-break: break-word; }
    textarea { width: 100%; height: 280px; }
    .refbox { background: #f8fafc; border: 1px solid #dbeafe; }
    .refimg { max-width: 100%; max-height: 520px; border: 1px solid #ddd; border-radius: 8px; display: block; margin-top: 10px; }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  <div class="card">
    <div class="stat">{{ labels.mode }}: {{ mode_name }}</div>
    <div class="stat">{{ labels.item }} {{ progress.answered + 1 }} / {{ progress.max_items }}</div>
    <div class="stat">Theta {{ '%.3f'|format(progress.theta) }}</div>
    <div class="stat">SE {{ '%.3f'|format(progress.se) }}</div>
    <div class="stat">{{ labels.choose_language }}: {{ language_name }}</div>
    <p class="muted">{{ progress.info_line }}</p>
  </div>

  <div class="card">
    <h2>{{ item.item_id }} · {{ labels.question }} {{ item.no }}</h2>
    <p style="white-space: pre-wrap;">{{ item.stem }}</p>
  </div>

  {% if item.link_href %}
  <div class="card refbox">
    <h3>{{ labels.figure_ref }}</h3>
    <p><a href="{{ item.link_href }}" target="_blank" rel="noopener">{{ labels.open_ref }}</a></p>
    {% if item.is_image_link %}
      <img class="refimg" src="{{ item.link_href }}" alt="Reference image for item {{ item.no }}">
    {% endif %}
  </div>
  {% endif %}

  <div class="card">
    <form method="post" action="{{ url_for('submit_answer') }}">
      {% for label, text in item.options.items() %}
      <label class="option">
        <span class="opt-row">
          <input class="opt-radio" type="radio" name="answer" value="{{ label }}" required>
          <span class="opt-label">{{ label }}</span>
          <span class="opt-text">{{ text }}</span>
        </span>
      </label>
      {% endfor %}
      {% if item.options|length == 0 %}
      <p class="muted">Choice parsing was not perfect, so the full item text is shown below. Select the best option label.</p>
      <textarea readonly>{{ item.full_text }}</textarea>
      <div style="margin-top:12px;">
        {% for label in ['A','B','C','D','E'] %}
          <label class="option">
            <span class="opt-row">
              <input class="opt-radio" type="radio" name="answer" value="{{ label }}" required>
              <span class="opt-label">{{ label }}</span>
              <span class="opt-text">Choose {{ label }}</span>
            </span>
          </label>
        {% endfor %}
      </div>
      {% endif %}
      <div style="margin-top:14px; display:flex; gap:12px; flex-wrap:wrap;">
        <button class="btn" type="submit">{{ labels.submit }}</button>
        <a class="btn" href="{{ url_for('to_home') }}">{{ labels.to_home }}</a>
      </div>
    </form>
  </div>
</body>
</html>
"""

VOICE_TMPL = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }}</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 1100px; margin: 24px auto; padding: 0 16px; line-height: 1.6; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 18px; margin-bottom: 18px; }
    .btn { display: inline-block; background: #2563eb; color: white; padding: 10px 16px; border-radius: 8px; text-decoration: none; border: 0; cursor: pointer; }
    .btn-secondary { background: #475569; }
    .btn.secondary { background: #475569; }
    .btn.warn { background: #b45309; }
    .stat { display: inline-block; margin-right: 16px; margin-bottom: 10px; padding: 10px 12px; background: #f7f7f7; border-radius: 10px; }
    .muted { color: #555; }
    .option { border: 1px solid #ddd; border-radius: 10px; padding: 10px 12px; margin: 10px 0; }
    .answer-box { display:none; margin-top: 14px; padding: 12px; border-radius: 10px; background: #eff6ff; border: 1px solid #bfdbfe; }
    .controls { display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
    .controls label { display:inline-flex; align-items:center; gap:6px; }
    .refbox { background: #f8fafc; border: 1px solid #dbeafe; }
    .refimg { max-width: 100%; max-height: 520px; border: 1px solid #ddd; border-radius: 8px; display: block; margin-top: 10px; }
    .audio-box { background: #f8fafc; border: 1px solid #dbeafe; }
    #audioPlayer { width: 100%; margin-top: 8px; }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  <div class="card">
    <div class="stat">{{ labels.mode }}: {{ labels.mode_voice_short }}</div>
    <div class="stat">{{ labels.voice_requested_items }}: {{ requested_max_items }}</div>
    <div class="stat">{{ labels.voice_actual_items }}: {{ item_count }}</div>
    <div class="stat">{{ labels.start_theta }} {{ '%.2f'|format(start_theta) }}</div>
    <div class="stat">{{ labels.voice_range_unit }} {{ '%.2f'|format(theta_range) }}</div>
    <div class="stat">{{ labels.choose_language }}: {{ language_name }}</div>
    <p class="muted">{{ labels.voice_intro }}</p>
    <p class="muted">{{ labels.voice_actual_note }}</p>
    <p class="muted" id="statusLine">{{ labels.voice_status_ready }}</p>
  </div>

  <div class="card audio-box">
    <strong>Audio player</strong>
    <audio id="audioPlayer" playsinline controls preload="none"></audio>
    <p class="muted" id="engineLine"></p>
    <p class="muted" id="audioUrlLine" style="word-break:break-all;"></p>
  </div>

  <div class="card">
    <h2 id="itemHeader"></h2>
    <p id="itemStem" style="white-space: pre-wrap;"></p>
    <div id="linkBox" class="card refbox" style="display:none; margin-top: 12px;">
      <h3>{{ labels.figure_ref }}</h3>
      <p><a id="linkHref" href="#" target="_blank" rel="noopener">{{ labels.open_ref }}</a></p>
      <img id="linkImg" class="refimg" src="" alt="Reference image" style="display:none;">
    </div>
    <div id="optionBox"></div>
    <div id="answerBox" class="answer-box">
      <strong>{{ labels.voice_correct }}:</strong> <span id="answerText"></span>
    </div>
  </div>

  <div class="card">
    <div class="controls">
      <button class="btn" type="button" onclick="startCycle()">{{ labels.voice_start_audio }}</button>
      <button class="btn secondary" type="button" onclick="replayCurrent()">{{ labels.voice_replay }}</button>
      <button class="btn secondary" type="button" onclick="pauseSpeech()">{{ labels.voice_pause }}</button>
      <button class="btn secondary" type="button" onclick="resumeSpeech()">{{ labels.voice_resume }}</button>
      <button class="btn warn" type="button" onclick="stopSpeech()">{{ labels.voice_stop }}</button>
      <button class="btn secondary" type="button" onclick="prevItem()">{{ labels.voice_prev }}</button>
      <button class="btn secondary" type="button" onclick="nextItem(false)">{{ labels.voice_next }}</button>
      <label><input type="checkbox" id="autoNext" checked> {{ labels.voice_auto_next }}</label>
      <label>{{ labels.voice_pause_seconds }} <input type="number" id="pauseSeconds" min="1" max="30" step="1" value="4"></label>
      <a class="btn secondary" href="{{ url_for('to_home') }}">{{ labels.to_home }}</a>
    </div>
  </div>

<script>
const items = {{ items|tojson }};
const speechConfig = {{ speech_config|tojson }};
const audioPlayer = document.getElementById('audioPlayer');
const isMobileLike = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent || '') || (window.matchMedia && window.matchMedia('(pointer: coarse)').matches) || ((navigator.maxTouchPoints || 0) > 0 && /MacIntel|Linux arm|Linux aarch64|Win32|Win64/i.test(navigator.platform || ''));
const requireManualStartOnEntry = !!isMobileLike;
const speechApi = ('speechSynthesis' in window) ? window.speechSynthesis : null;
const hasBrowserTTS = !!speechApi && (typeof SpeechSynthesisUtterance !== 'undefined');
let currentIndex = 0;
let betweenTimer = null;
let nextTimer = null;
let didAutoStart = false;
let currentObjectUrl = null;
let activeEngine = 'idle';
let currentMode = 'server';

function setStatus(text) {
  document.getElementById('statusLine').textContent = text || '';
}

function setEngine(text) {
  document.getElementById('engineLine').textContent = text || '';
}

function clearTimers() {
  if (betweenTimer) { clearTimeout(betweenTimer); betweenTimer = null; }
  if (nextTimer) { clearTimeout(nextTimer); nextTimer = null; }
}

function clearAudioObjectUrl() {
  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = null;
  }
}

function resetAudioPlayer() {
  audioPlayer.pause();
  audioPlayer.onended = null;
  audioPlayer.onerror = null;
  clearAudioObjectUrl();
  audioPlayer.removeAttribute('src');
  audioPlayer.load();
  setAudioUrl('');
}

function stopSpeech() {
  clearTimers();
  if (speechApi && speechApi.cancel) speechApi.cancel();
  resetAudioPlayer();
  activeEngine = 'idle';
  setStatus(speechConfig.stopped_status);
}

function pauseSpeech() {
  clearTimers();
  if (activeEngine === 'server') {
    audioPlayer.pause();
  } else {
    if (speechApi && speechApi.pause) speechApi.pause();
  }
  setStatus(speechConfig.paused_status);
}

function resumeSpeech() {
  if (activeEngine === 'server') {
    audioPlayer.play().then(() => {
      setStatus(speechConfig.resumed_status);
    }).catch((err) => {
      setStatus(`${speechConfig.server_failed_status} ${err && err.message ? err.message : err}`);
    });
  } else {
    if (speechApi && speechApi.resume) speechApi.resume();
    setStatus(speechConfig.resumed_status);
  }
}

function renderItem() {
  const item = items[currentIndex];
  if (!item) {
    document.getElementById('itemHeader').textContent = speechConfig.done_title;
    document.getElementById('itemStem').textContent = speechConfig.done_body;
    document.getElementById('optionBox').innerHTML = '';
    document.getElementById('answerBox').style.display = 'none';
    document.getElementById('linkBox').style.display = 'none';
    setStatus(speechConfig.done_body);
    return;
  }
  document.getElementById('itemHeader').textContent = `${item.item_id} · ${speechConfig.item_label} ${currentIndex + 1} / ${items.length}`;
  document.getElementById('itemStem').textContent = item.stem || item.full_text || '';
  const optionBox = document.getElementById('optionBox');
  optionBox.innerHTML = '';
  (item.options || []).forEach(opt => {
    const div = document.createElement('div');
    div.className = 'option';
    div.textContent = `${opt.label}. ${opt.text}`;
    optionBox.appendChild(div);
  });
  const answerBox = document.getElementById('answerBox');
  answerBox.style.display = 'none';
  document.getElementById('answerText').textContent = `${item.key}${item.answer_text ? '. ' + item.answer_text : ''}`;
  const linkBox = document.getElementById('linkBox');
  const linkHref = document.getElementById('linkHref');
  const linkImg = document.getElementById('linkImg');
  if (item.link_href) {
    linkBox.style.display = 'block';
    linkHref.href = item.link_href;
    if (item.is_image_link) {
      linkImg.style.display = 'block';
      linkImg.src = item.link_href;
    } else {
      linkImg.style.display = 'none';
      linkImg.src = '';
    }
  } else {
    linkBox.style.display = 'none';
    linkHref.href = '#';
    linkImg.style.display = 'none';
    linkImg.src = '';
  }
  setStatus(`${speechConfig.ready_status} ${currentIndex + 1} / ${items.length}`);
}

function revealAnswer() {
  document.getElementById('answerBox').style.display = 'block';
}

function setAudioUrl(text) {
  const el = document.getElementById('audioUrlLine');
  if (el) el.textContent = text || '';
}

function chooseBestVoice() {
  const voices = (speechApi && speechApi.getVoices) ? speechApi.getVoices() : [];
  if (!voices || !voices.length) return null;
  const langPrefix = (speechConfig.lang || '').toLowerCase();
  return voices.find(v => (v.lang || '').toLowerCase() === langPrefix)
      || voices.find(v => (v.lang || '').toLowerCase().startsWith(langPrefix.split('-')[0]))
      || voices[0];
}

function speakParts(parts, doneCallback) {
  clearTimers();
  resetAudioPlayer();
  setAudioUrl('');
  if (speechApi && speechApi.cancel) speechApi.cancel();
  const queue = (parts || []).map(x => (x || '').trim()).filter(Boolean);
  activeEngine = 'browser';
  currentMode = 'browser';
  setEngine(speechConfig.browser_engine_label);
  if (!hasBrowserTTS) {
    setStatus(speechConfig.browser_unavailable_status || 'Browser TTS unavailable on this device.');
    if (doneCallback) doneCallback();
    return;
  }
  function next() {
    if (!queue.length) {
      if (doneCallback) doneCallback();
      return;
    }
    const utter = new SpeechSynthesisUtterance(queue.shift());
    utter.lang = speechConfig.lang;
    utter.rate = 1.0;
    utter.pitch = 1.0;
    const bestVoice = chooseBestVoice();
    if (bestVoice) utter.voice = bestVoice;
    utter.onend = next;
    utter.onerror = next;
    if (speechApi && speechApi.speak) { speechApi.speak(utter); } else { next(); }
  }
  next();
}

function currentQuestionParts(item) {
  const parts = [];
  const qLead = speechConfig.lang.startsWith('zh') ? `${speechConfig.question_prefix}${currentIndex + 1}題。` : `${speechConfig.question_prefix} ${currentIndex + 1}.`;
  parts.push(`${qLead} ${item.stem || item.full_text || ''}`);
  (item.options || []).forEach(opt => parts.push(`${opt.label}. ${opt.text}`));
  return parts;
}

function currentAnswerParts(item) {
  const answerLine = item.answer_text ? `${speechConfig.answer_prefix} ${item.key}. ${item.answer_text}` : `${speechConfig.answer_prefix} ${item.key}.`;
  return [answerLine];
}

function currentVoiceUrl(part) {
  const item = items[currentIndex];
  if (!item) return '';
  const params = new URLSearchParams({
    item_id: item.item_id,
    part: part,
    lang: speechConfig.language_code || 'en',
    n: String(currentIndex + 1),
    _: String(Date.now())
  });
  return `${speechConfig.tts_base_url}?${params.toString()}`;
}

async function playServerUrl(url, onEnded) {
  resetAudioPlayer();
  setAudioUrl(url);
  audioPlayer.src = url;
  audioPlayer.onended = () => {
    audioPlayer.onended = null;
    if (onEnded) onEnded();
  };
  audioPlayer.onerror = () => {
    audioPlayer.onerror = null;
    setStatus(`${speechConfig.server_failed_status} ${url}`);
    startBrowserCycle();
  };
  activeEngine = 'server';
  currentMode = 'server';
  setEngine(speechConfig.server_engine_label);
  await audioPlayer.play();
}

function finishItemAfterAnswer() {
  setStatus(`${speechConfig.finished_status} ${currentIndex + 1} / ${items.length}`);
  if (document.getElementById('autoNext').checked) {
    nextTimer = setTimeout(() => nextItem(true), 1800);
  }
}

function startBrowserCycle() {
  const item = items[currentIndex];
  if (!item) {
    setStatus(speechConfig.done_body);
    return;
  }
  renderItem();
  setStatus(`${speechConfig.reading_status} ${currentIndex + 1} / ${items.length}`);
  speakParts(currentQuestionParts(item), () => {
    const pauseMs = Math.max(1000, (parseFloat(document.getElementById('pauseSeconds').value) || 4) * 1000);
    setStatus(speechConfig.pause_status);
    betweenTimer = setTimeout(() => {
      revealAnswer();
      setStatus(speechConfig.answer_status);
      speakParts(currentAnswerParts(item), () => finishItemAfterAnswer());
    }, pauseMs);
  });
}

async function startServerCycle() {
  const item = items[currentIndex];
  if (!item) {
    setStatus(speechConfig.done_body);
    return;
  }
  renderItem();
  setStatus(`${speechConfig.reading_status} ${currentIndex + 1} / ${items.length}`);
  try {
    const qUrl = currentVoiceUrl('question');
    await playServerUrl(qUrl, () => {
      const pauseMs = Math.max(1000, (parseFloat(document.getElementById('pauseSeconds').value) || 4) * 1000);
      setStatus(speechConfig.pause_status);
      betweenTimer = setTimeout(async () => {
        revealAnswer();
        setStatus(speechConfig.answer_status);
        try {
          const aUrl = currentVoiceUrl('answer');
          await playServerUrl(aUrl, () => finishItemAfterAnswer());
        } catch (err) {
          setStatus(`${speechConfig.server_failed_status} ${err && err.message ? err.message : err}`);
          speakParts(currentAnswerParts(item), () => finishItemAfterAnswer());
        }
      }, pauseMs);
    });
  } catch (err) {
    setStatus(`${speechConfig.server_failed_status} ${err && err.message ? err.message : err}`);
    startBrowserCycle();
  }
}

function startCycle() {
  stopSpeech();
  currentMode = speechConfig.server_tts_enabled ? 'server' : 'browser';
  if (currentMode === 'server') {
    startServerCycle();
  } else {
    startBrowserCycle();
  }
}

function replayCurrent() {
  startCycle();
}

function prevItem() {
  stopSpeech();
  if (currentIndex > 0) currentIndex -= 1;
  renderItem();
}

function nextItem(autoStart) {
  stopSpeech();
  currentIndex += 1;
  if (currentIndex >= items.length) {
    currentIndex = items.length;
    renderItem();
    return;
  }
  renderItem();
  if (autoStart) startCycle();
}

function bindMobileButtons() {
  if (!isMobileLike) return;
  document.querySelectorAll('button[onclick]').forEach((btn) => {
    const attr = btn.getAttribute('onclick') || '';
    btn.addEventListener('touchend', (ev) => {
      ev.preventDefault();
      try {
        if (attr.includes('startCycle')) startCycle();
        else if (attr.includes('replayCurrent')) replayCurrent();
        else if (attr.includes('pauseSpeech')) pauseSpeech();
        else if (attr.includes('resumeSpeech')) resumeSpeech();
        else if (attr.includes('stopSpeech')) stopSpeech();
        else if (attr.includes('prevItem')) prevItem();
        else if (attr.includes('nextItem')) nextItem(false);
      } catch (err) {
        setStatus(`${speechConfig.touch_failed_status || 'Touch action failed.'} ${err && err.message ? err.message : err}`);
      }
    }, {passive:false});
  });
}

window.addEventListener('beforeunload', () => {
  if (speechApi && speechApi.cancel) speechApi.cancel();
  clearTimers();
  resetAudioPlayer();
});

window.addEventListener('load', () => {
  renderItem();
  bindMobileButtons();
  if (speechConfig.server_tts_enabled) {
    setEngine(speechConfig.server_engine_label);
    setStatus(requireManualStartOnEntry ? (speechConfig.mobile_tap_status || 'On mobile, press Start audio cycle to begin.') : speechConfig.ready_status);
  } else {
    setEngine(speechConfig.browser_engine_label);
    if (!hasBrowserTTS) setStatus(speechConfig.browser_unavailable_status || 'Browser TTS unavailable on this device.');
    else setStatus(requireManualStartOnEntry ? (speechConfig.mobile_tap_status || 'On mobile, press Start audio cycle to begin.') : speechConfig.ready_status);
  }
  if (!didAutoStart && items.length > 0 && !requireManualStartOnEntry) {
    didAutoStart = true;
    setTimeout(() => startCycle(), 300);
  }
});
</script>
</body>
</html>
"""

RESULT_TMPL = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }}</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 1100px; margin: 24px auto; padding: 0 16px; line-height: 1.6; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 18px; margin-bottom: 18px; }
    .btn { display: inline-block; background: #2563eb; color: white; padding: 10px 16px; border-radius: 8px; text-decoration: none; border: 0; cursor: pointer; }
    .btn-secondary { background: #475569; }
    .stat { display: inline-block; margin-right: 16px; margin-bottom: 10px; padding: 10px 12px; background: #f7f7f7; border-radius: 10px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }
    .muted { color: #555; }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  <div class="card">
    <div class="stat">{{ labels.mode }} {{ result.mode_name }}</div>
    <div class="stat">{{ labels.final_theta }} {{ '%.3f'|format(result.theta) }}</div>
    <div class="stat">{{ labels.post_se }} {{ '%.3f'|format(result.se) }}</div>
    <div class="stat">{{ labels.percentile }} {{ '%.1f'|format(result.percentile) }}</div>
    <div class="stat">{{ labels.items_used }} {{ result.n_answered }}</div>
    <div class="stat">{{ labels.reason }} {{ result.stop_reason }}</div>
  </div>
  <div class="card">
    <div class="stat">{{ labels.infit_mnsq }} {{ '%.3f'|format(result.infit_mnsq) }}</div>
    <div class="stat">{{ labels.outfit_mnsq }} {{ '%.3f'|format(result.outfit_mnsq) }}</div>
    <p class="muted">{{ labels.answer_hidden }}</p>
    <p class="muted">CAT commonly targets items near the current theta. When model probabilities stay near 0.5, even random dichotomous answers can leave INFIT/OUTFIT close to 1.0; that is not necessarily a coding error.</p>
    <div style="margin-top:12px;"><a class="btn" href="{{ url_for('index') }}">{{ labels.new_test }}</a></div>
  </div>
  <div class="card">
    <h2>{{ labels.responses }}</h2>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>{{ labels.item }}</th>
          <th>{{ labels.difficulty }}</th>
          <th>{{ labels.your_answer }}</th>
          <th>{{ labels.answer_match }}</th>
          <th>{{ labels.theta_after }}</th>
          <th>{{ labels.se_after }}</th>
          <th>{{ labels.zstd }}</th>
          {% if result.has_links %}<th>{{ labels.reference }}</th>{% endif %}
        </tr>
      </thead>
      <tbody>
      {% for row in result.history %}
        <tr>
          <td>{{ loop.index }}</td>
          <td>{{ row.item_id }}</td>
          <td>{{ '%.3f'|format(row.delta) }}</td>
          <td>{{ row.answer }}</td>
          <td>{{ row.correct_tf }}</td>
          <td>{{ '%.3f'|format(row.theta) }}</td>
          <td>{{ '%.3f'|format(row.se) }}</td>
          <td>{{ '%.3f'|format(row.zstd if row.zstd is defined else 0) }}</td>
          {% if result.has_links %}
            <td>{% if row.link_href %}<a href="{{ row.link_href }}" target="_blank" rel="noopener">{{ labels.open }}</a>{% endif %}</td>
          {% endif %}
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>{{ labels.trend_chart }}</h2>
    <div>{{ result.trend_svg|safe }}</div>
  </div>

  <div class="card">
    <h2>{{ labels.kidmap }}</h2>
    <div>{{ result.kidmap_svg|safe }}</div>
    <p class="muted">{{ labels.kidmap_note }}</p>
    <p class="muted">{{ labels.answered_items }}: {{ result.answered_ids|join(', ') }}</p>
  </div>



  {% if result.comparison %}
  <div class="card">
    <h2>CAT vs non-CAT comparison (SE-based stopping)</h2>
    <div class="stat">n (= simulated CAT number) {{ result.comparison.n_reps }}</div>
    <div class="stat">Selected person {{ result.comparison.person_id }}</div>
    <div class="stat">Full non-CAT length {{ result.comparison.full_length }}</div>
    <div class="stat">CAT stopping rule posterior SE ≤ {{ '%.3f'|format(result.comparison.stop_se) }}</div>
    <div class="stat">Mean CAT length {{ '%.3f'|format(result.comparison.cat_length_summary.mean) }}</div>
    <div class="stat">CAT length SD {{ '%.3f'|format(result.comparison.cat_length_summary.sd) }}</div>

    <h3>Table 1. Item length</h3>
    <table>
      <thead>
        <tr>
          <th>Group</th><th>n</th><th>Mean length</th><th>SD length</th><th>Min</th><th>Median</th><th>Max</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Full non-CAT</td>
          <td>{{ result.comparison.full_summary.n }}</td>
          <td>{{ '%.3f'|format(result.comparison.full_summary.mean_length) }}</td>
          <td>{{ '%.3f'|format(result.comparison.full_summary.sd_length) }}</td>
          <td>{{ '%.3f'|format(result.comparison.full_length_stats.min) }}</td>
          <td>{{ '%.3f'|format(result.comparison.full_length_stats.med) }}</td>
          <td>{{ '%.3f'|format(result.comparison.full_length_stats.max) }}</td>
        </tr>
        <tr>
          <td>CAT</td>
          <td>{{ result.comparison.n_reps }}</td>
          <td>{{ '%.3f'|format(result.comparison.cat_length_summary.mean) }}</td>
          <td>{{ '%.3f'|format(result.comparison.cat_length_summary.sd) }}</td>
          <td>{{ '%.3f'|format(result.comparison.cat_length_stats.min) }}</td>
          <td>{{ '%.3f'|format(result.comparison.cat_length_stats.med) }}</td>
          <td>{{ '%.3f'|format(result.comparison.cat_length_stats.max) }}</td>
        </tr>
      </tbody>
    </table>
    <p class="muted" style="margin-top:10px;">One-sample t-test of CAT item length against the full non-CAT length ({{ result.comparison.full_length }} items): t = {{ result.comparison.length_ttest.t_text }}, p = {{ result.comparison.length_ttest.p_text }}, df = {{ result.comparison.length_ttest.df_text }}.</p>
    <p class="muted">{{ result.comparison.length_ttest.note }}</p>
    <div style="margin-top:12px;">{{ result.comparison.length_svg|safe }}</div>

    <h3>Table 2. Person measure</h3>
    <table>
      <thead>
        <tr>
          <th>Group</th><th>n</th><th>Mean theta</th><th>SD theta</th><th>Min</th><th>Median</th><th>Max</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Full non-CAT</td>
          <td>{{ result.comparison.n_reps }}</td>
          <td>{{ '%.3f'|format(result.comparison.full_theta_summary.mean) }}</td>
          <td>{{ '%.3f'|format(result.comparison.full_theta_summary.sd) }}</td>
          <td>{{ '%.3f'|format(result.comparison.full_theta_stats.min) }}</td>
          <td>{{ '%.3f'|format(result.comparison.full_theta_stats.med) }}</td>
          <td>{{ '%.3f'|format(result.comparison.full_theta_stats.max) }}</td>
        </tr>
        <tr>
          <td>CAT</td>
          <td>{{ result.comparison.n_reps }}</td>
          <td>{{ '%.3f'|format(result.comparison.cat_theta_summary.mean) }}</td>
          <td>{{ '%.3f'|format(result.comparison.cat_theta_summary.sd) }}</td>
          <td>{{ '%.3f'|format(result.comparison.cat_theta_stats.min) }}</td>
          <td>{{ '%.3f'|format(result.comparison.cat_theta_stats.med) }}</td>
          <td>{{ '%.3f'|format(result.comparison.cat_theta_stats.max) }}</td>
        </tr>
      </tbody>
    </table>
    <p class="muted" style="margin-top:10px;">One-sample t-test of CAT person measure against the selected person's full non-CAT measure: t = {{ result.comparison.theta_diff_ttest.t_text }}, p = {{ result.comparison.theta_diff_ttest.p_text }}, df = {{ result.comparison.theta_diff_ttest.df_text }}. Mean difference (CAT − full non-CAT) = {{ '%.3f'|format(result.comparison.theta_diff_summary.mean) }}, SD difference = {{ '%.3f'|format(result.comparison.theta_diff_summary.sd) }}.</p>
    <div style="margin-top:12px;">{{ result.comparison.theta_svg|safe }}</div>

    <p class="muted">Both figures are box plots. The left compares CAT and full non-CAT by item length; the right compares CAT and full non-CAT by person measure. In this section, one person is randomly selected once from original_response.csv, full non-CAT is computed once from that person's complete responses, and CAT is rerun n times using the same person's item responses with a random first item. CAT stopping is governed by posterior SE, with the full bank length used only as a safety cap.</p>
    <h3>Example CAT administration used for the CPC / KIDMAP shown above</h3>
    <table>
      <thead>
        <tr><th>Pos</th><th>Item</th><th>No</th><th>Delta</th><th>Reference</th></tr>
      </thead>
      <tbody>
        {% for row in result.comparison.actual_rows %}
        <tr>
          <td>{{ row.pos }}</td>
          <td>{{ row.item_id }}</td>
          <td>{{ row.no }}</td>
          <td>{{ '%.3f'|format(row.delta) }}</td>
          <td>{% if row.link_href %}<a href="{{ row.link_href }}" target="_blank" rel="noopener">{{ labels.open }}</a>{% endif %}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
  {% endif %}

  {% if result.comparison_n %}
  <div class="card">
    <h2>CAT vs non-CAT(n) comparison (n persons from original_response.csv)</h2>
    <div class="stat">n (= selected persons) {{ result.comparison_n.n_persons }}</div>
    <div class="stat">Selected persons {{ result.comparison_n.selected_person_preview }}</div>
    <div class="stat">CAT stopping rule posterior SE ≤ {{ '%.3f'|format(result.comparison_n.stop_se) }}</div>
    <div class="stat">Mean full non-CAT length {{ '%.3f'|format(result.comparison_n.full_summary.mean_length) }}</div>
    <div class="stat">Mean CAT length {{ '%.3f'|format(result.comparison_n.cat_length_summary.mean) }}</div>

    <h3>Table 1. Item length</h3>
    <table>
      <thead>
        <tr>
          <th>Group</th><th>n</th><th>Mean length</th><th>SD length</th><th>Min</th><th>Median</th><th>Max</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Full non-CAT</td>
          <td>{{ result.comparison_n.full_summary.n }}</td>
          <td>{{ '%.3f'|format(result.comparison_n.full_summary.mean_length) }}</td>
          <td>{{ '%.3f'|format(result.comparison_n.full_summary.sd_length) }}</td>
          <td>{{ '%.3f'|format(result.comparison_n.full_length_stats.min) }}</td>
          <td>{{ '%.3f'|format(result.comparison_n.full_length_stats.med) }}</td>
          <td>{{ '%.3f'|format(result.comparison_n.full_length_stats.max) }}</td>
        </tr>
        <tr>
          <td>CAT</td>
          <td>{{ result.comparison_n.n_persons }}</td>
          <td>{{ '%.3f'|format(result.comparison_n.cat_length_summary.mean) }}</td>
          <td>{{ '%.3f'|format(result.comparison_n.cat_length_summary.sd) }}</td>
          <td>{{ '%.3f'|format(result.comparison_n.cat_length_stats.min) }}</td>
          <td>{{ '%.3f'|format(result.comparison_n.cat_length_stats.med) }}</td>
          <td>{{ '%.3f'|format(result.comparison_n.cat_length_stats.max) }}</td>
        </tr>
      </tbody>
    </table>
    <p class="muted" style="margin-top:10px;">Paired t-test of CAT item length versus full non-CAT item length across the same selected persons: t = {{ result.comparison_n.length_ttest.t_text }}, p = {{ result.comparison_n.length_ttest.p_text }}, df = {{ result.comparison_n.length_ttest.df_text }}. Mean difference (CAT − full non-CAT) = {{ '%.3f'|format(result.comparison_n.length_ttest.mean) }}, SD difference = {{ '%.3f'|format(result.comparison_n.length_ttest.sd) }}.</p>
    <div style="margin-top:12px;">{{ result.comparison_n.length_svg|safe }}</div>

    <h3>Table 2. Person measure</h3>
    <table>
      <thead>
        <tr>
          <th>Group</th><th>n</th><th>Mean theta</th><th>SD theta</th><th>Min</th><th>Median</th><th>Max</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Full non-CAT</td>
          <td>{{ result.comparison_n.full_theta_summary.n }}</td>
          <td>{{ '%.3f'|format(result.comparison_n.full_theta_summary.mean) }}</td>
          <td>{{ '%.3f'|format(result.comparison_n.full_theta_summary.sd) }}</td>
          <td>{{ '%.3f'|format(result.comparison_n.full_theta_stats.min) }}</td>
          <td>{{ '%.3f'|format(result.comparison_n.full_theta_stats.med) }}</td>
          <td>{{ '%.3f'|format(result.comparison_n.full_theta_stats.max) }}</td>
        </tr>
        <tr>
          <td>CAT</td>
          <td>{{ result.comparison_n.n_persons }}</td>
          <td>{{ '%.3f'|format(result.comparison_n.cat_theta_summary.mean) }}</td>
          <td>{{ '%.3f'|format(result.comparison_n.cat_theta_summary.sd) }}</td>
          <td>{{ '%.3f'|format(result.comparison_n.cat_theta_stats.min) }}</td>
          <td>{{ '%.3f'|format(result.comparison_n.cat_theta_stats.med) }}</td>
          <td>{{ '%.3f'|format(result.comparison_n.cat_theta_stats.max) }}</td>
        </tr>
      </tbody>
    </table>
    <p class="muted" style="margin-top:10px;">Paired t-test of CAT person measure versus full non-CAT person measure across the same selected persons: t = {{ result.comparison_n.theta_diff_ttest.t_text }}, p = {{ result.comparison_n.theta_diff_ttest.p_text }}, df = {{ result.comparison_n.theta_diff_ttest.df_text }}. Mean difference (CAT − full non-CAT) = {{ '%.3f'|format(result.comparison_n.theta_diff_summary.mean) }}, SD difference = {{ '%.3f'|format(result.comparison_n.theta_diff_summary.sd) }}.</p>
    <div style="margin-top:12px;">{{ result.comparison_n.theta_svg|safe }}</div>

    <p class="muted">Both figures are box plots. The first compares CAT and full non-CAT by item length; the second compares CAT and full non-CAT by person measure. In this additional CAT vs non-CAT(n) section, n persons are sampled once from original_response.csv using the homepage setting of Simulated CAT number (comparison). Each selected person contributes one full non-CAT run using all available item responses and one CAT run using the same person responses with a random first item and SE-based stopping.</p>
  </div>
  {% endif %}

  <div class="card">
    <h2>Category probability curves (CPC)</h2>
    <div>{{ result.cpc_svg|safe }}</div>
    <p class="muted">This CPC assumes a <strong>reference item with delta = 0</strong>. The red dotted vertical line marks the curve intersection at theta = 0, and the solid red vertical line marks the final person measure.</p>
    <p class="muted">At your final theta, the more probable response on the delta = 0 reference item is <strong>{{ result.cpc_pred_label }}</strong>.</p>
  </div>

  <div class="card">
    <h2>{{ labels.item_fit }}</h2>
    <div>{{ result.itemfit_svg|safe }}</div>
    <p class="muted">{{ labels.itemfit_note }}</p>
  </div>
</body>
</html>
"""


@dataclass
class ItemRecord:
    item_id: str
    no: int
    key: str
    full_text_zh: str
    stem_zh: str
    options_zh: Dict[str, str]
    full_text_en: str
    stem_en: str
    options_en: Dict[str, str]
    delta: float
    link: str = ""

    def text_for(self, language: str) -> Dict[str, object]:
        if language == "en":
            return {
                "full_text": self.full_text_en or self.full_text_zh,
                "stem": self.stem_en or self.stem_zh,
                "options": self.options_en or self.options_zh,
            }
        return {
            "full_text": self.full_text_zh,
            "stem": self.stem_zh,
            "options": self.options_zh,
        }


def _zip_name_map(zf: zipfile.ZipFile) -> Dict[str, str]:
    return {name.lower(): name for name in zf.namelist()}


def _zip_read_bytes(zf: zipfile.ZipFile, wanted_name: str) -> bytes:
    name_map = _zip_name_map(zf)
    hit = name_map.get(wanted_name.lower())
    if not hit:
        raise KeyError(f"{wanted_name} not found in ZIP. Available: {zf.namelist()}")
    return zf.read(hit)


def _read_csv_bytes_robust(raw: bytes, *, csv_name: str = "csv") -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp950", "big5", "gb18030", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
        except pd.errors.ParserError:
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc, engine="python")
            except Exception as e:
                last_err = e
        except Exception as e:
            last_err = e
    raise ValueError(f"Unable to read {csv_name}. Tried encodings: {encodings}. Last error: {last_err}")


def _read_text_bytes_robust(raw: bytes, *, text_name: str = "text") -> str:
    encodings = ["utf-8", "utf-8-sig", "cp950", "big5", "gb18030", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError as e:
            last_err = e
    raise ValueError(f"Unable to decode {text_name}. Tried encodings: {encodings}. Last error: {last_err}")




def _trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


class RaschCATBank:
    def __init__(self, bundle_path: Path) -> None:
        self.bundle_path = bundle_path
        self.extract_dir = Path(os.environ.get("TMPDIR") or tempfile.gettempdir()) / "raschcatskin_bundle_cache"
        self.local_pic_dir = Path(__file__).with_name("pic")
        self.readme_text = README_FALLBACK
        self.items: List[ItemRecord] = []
        self.item_lookup: Dict[str, ItemRecord] = {}
        self.theta_grid = np.linspace(-6.0, 6.0, 2401)
        self.prior_mean = 0.0
        self.prior_sd = 1.0
        self.person_distribution = np.array([0.0])
        self.person_df = pd.DataFrame()
        self.item_fit_df = pd.DataFrame()
        self.original_response_df = pd.DataFrame()
        self.original_response_item_ids: List[str] = []
        self.original_response_person_col = "person_id"
        self._load()

    def _extract_selected_files(self, zf: zipfile.ZipFile) -> None:
        self.extract_dir.mkdir(parents=True, exist_ok=True)
        needed = []
        for name in zf.namelist():
            low = name.lower()
            if low in {"response_category.csv", "fixed_item_delta.csv", "person_estimates.csv", "item_estimates.csv", "original_response.csv", "metadata.json", "readme.md"} or low.startswith("pic/"):
                needed.append(name)
        for name in needed:
            target = self.extract_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(target, "wb") as dst:
                dst.write(src.read())

    def _load(self) -> None:
        if not self.bundle_path.exists():
            raise FileNotFoundError(f"Bundle not found: {self.bundle_path}")
        with zipfile.ZipFile(self.bundle_path, "r") as zf:
            self._extract_selected_files(zf)
            response_df = _read_csv_bytes_robust(_zip_read_bytes(zf, "response_category.csv"), csv_name="response_category.csv")
            delta_df = _read_csv_bytes_robust(_zip_read_bytes(zf, "fixed_item_delta.csv"), csv_name="fixed_item_delta.csv")
            person_df = _read_csv_bytes_robust(_zip_read_bytes(zf, "person_estimates.csv"), csv_name="person_estimates.csv")
            item_fit_df = _read_csv_bytes_robust(_zip_read_bytes(zf, "item_estimates.csv"), csv_name="item_estimates.csv")
            metadata = json.loads(_read_text_bytes_robust(_zip_read_bytes(zf, "metadata.json"), text_name="metadata.json"))
            name_map = _zip_name_map(zf)
            original_response_df = pd.DataFrame()
            if "original_response.csv" in name_map:
                original_response_df = _read_csv_bytes_robust(zf.read(name_map["original_response.csv"]), csv_name="original_response.csv")
            if "readme.md" in name_map:
                self.readme_text = _read_text_bytes_robust(zf.read(name_map["readme.md"]), text_name="README.md")

        response_df = response_df.copy()
        expected_cols = ["key", "no", "link", "item", "item2"]
        for col in expected_cols:
            if col not in response_df.columns:
                response_df[col] = ""
        response_df["link"] = response_df["link"].fillna("").astype(str).str.strip()
        response_df["item"] = response_df["item"].fillna("").astype(str)
        response_df["item2"] = response_df["item2"].fillna("").astype(str)

        delta_df = delta_df.copy()
        item_labels = delta_df["ITEM"].astype(str).str.strip()
        extracted_no = item_labels.str.extract(r"(\d+)(?!.*\d)", expand=False)
        delta_df["no"] = pd.to_numeric(extracted_no, errors="coerce")
        delta_df = delta_df.dropna(subset=["no"]).copy()
        delta_df["no"] = delta_df["no"].astype(int)
        delta_df["DELTA"] = pd.to_numeric(delta_df["DELTA"], errors="coerce")

        # Avoid exploding each item into repeated rows (for example 250 items × 5 category rows = 1250).
        # If the delta file is already row-aligned with response_category.csv, keep that one-to-one alignment.
        # Otherwise reduce delta_df to one row per item number before merging.
        if len(delta_df) == len(response_df):
            merged = response_df.reset_index(drop=True).copy()
            merged["ITEM"] = delta_df["ITEM"].reset_index(drop=True)
            merged["DELTA"] = delta_df["DELTA"].reset_index(drop=True)
        else:
            delta_one = (
                delta_df.sort_values(["no"])
                .groupby("no", as_index=False)
                .agg({"ITEM": "first", "DELTA": "mean"})
            )
            merged = response_df.merge(delta_one[["no", "ITEM", "DELTA"]], on="no", how="left")

        merged["ITEM"] = merged["ITEM"].fillna(merged["no"].map(lambda x: f"Q{x:03d}"))
        merged["DELTA"] = pd.to_numeric(merged["DELTA"], errors="coerce")
        med_delta = float(np.nanmedian(merged["DELTA"].to_numpy(dtype=float))) if np.isfinite(np.nanmedian(merged["DELTA"].to_numpy(dtype=float))) else 0.0
        merged["DELTA"] = merged["DELTA"].fillna(med_delta)

        self.person_df = person_df.copy()
        self.item_fit_df = item_fit_df.copy()
        measures = pd.to_numeric(person_df.get("MEASURE"), errors="coerce").dropna().to_numpy()
        if measures.size > 5:
            self.prior_mean = float(np.mean(measures))
            self.prior_sd = float(np.std(measures, ddof=1))
            self.person_distribution = measures
        self.prior_sd = max(self.prior_sd, 0.5)
        self.model = metadata.get("model", "Rasch")

        items: List[ItemRecord] = []
        for row in merged.itertuples(index=False):
            stem_zh, options_zh = parse_item_text(str(row.item))
            english_text = str(row.item2).strip() or str(row.item)
            stem_en, options_en = parse_item_text(english_text)
            rec = ItemRecord(
                item_id=str(row.ITEM),
                no=int(row.no),
                key=str(row.key).strip().upper(),
                full_text_zh=str(row.item),
                stem_zh=stem_zh,
                options_zh=options_zh,
                full_text_en=english_text,
                stem_en=stem_en,
                options_en=options_en,
                delta=float(row.DELTA),
                link=str(getattr(row, "link", "") or "").strip(),
            )
            items.append(rec)
        items.sort(key=lambda x: x.no)
        self.items = items
        self.item_lookup = {x.item_id: x for x in items}
        item_ids = [item.item_id for item in items]
        unique_item_ids = list(dict.fromkeys(item_ids))
        if not original_response_df.empty:
            person_col = "person_id" if "person_id" in original_response_df.columns else ("KID" if "KID" in original_response_df.columns else str(original_response_df.columns[0]))
            keep_cols = [person_col] + [c for c in unique_item_ids if c in original_response_df.columns]
            for extra_col in ["Profile", "profile"]:
                if extra_col in original_response_df.columns and extra_col not in keep_cols:
                    keep_cols.append(extra_col)
            original_response_df = original_response_df.loc[:, keep_cols].copy()
            if original_response_df.columns.duplicated().any():
                original_response_df = original_response_df.loc[:, ~original_response_df.columns.duplicated()].copy()
            for c in unique_item_ids:
                if c in original_response_df.columns:
                    col = original_response_df[c]
                    if isinstance(col, pd.DataFrame):
                        for subcol in col.columns:
                            original_response_df[subcol] = pd.to_numeric(original_response_df[subcol], errors="coerce")
                    else:
                        original_response_df[c] = pd.to_numeric(col, errors="coerce")
            score_cols = [c for c in unique_item_ids if c in original_response_df.columns]
            if score_cols:
                original_response_df = original_response_df.dropna(subset=score_cols, how="all").reset_index(drop=True)
            self.original_response_df = original_response_df
            self.original_response_item_ids = score_cols
            self.original_response_person_col = person_col

    def probability(self, theta: np.ndarray | float, delta: float) -> np.ndarray | float:
        x = np.clip(np.asarray(theta) - delta, -35, 35)
        return 1.0 / (1.0 + np.exp(-x))

    def information(self, theta: float, delta: float) -> float:
        p = float(self.probability(theta, delta))
        return p * (1.0 - p)

    def posterior(self, responses: List[Tuple[str, int]], start_theta: float | None = None) -> Tuple[float, float, np.ndarray]:
        grid = self.theta_grid
        mu = self.prior_mean if start_theta is None else float(start_theta)
        sd = self.prior_sd
        log_post = -0.5 * ((grid - mu) / sd) ** 2 - np.log(sd * math.sqrt(2.0 * math.pi))
        for item_id, score in responses:
            delta = self.item_lookup[item_id].delta
            p = np.clip(self.probability(grid, delta), 1e-12, 1 - 1e-12)
            log_post += np.log(p) if score == 1 else np.log(1.0 - p)
        log_post -= np.max(log_post)
        post = np.exp(log_post)
        den = _trapz_compat(post, grid)
        if not np.isfinite(den) or den <= 0:
            den = 1.0
        post = post / den
        mean = _trapz_compat(grid * post, grid)
        var = _trapz_compat(((grid - mean) ** 2) * post, grid)
        se = max(math.sqrt(max(var, 1e-10)), 1e-6)
        return mean, se, post

    def select_next_item(self, administered: List[str], theta: float) -> ItemRecord:
        used = set(administered)
        remaining = [item for item in self.items if item.item_id not in used]
        if not remaining:
            raise RuntimeError("No remaining items.")
        return max(remaining, key=lambda item: (self.information(theta, item.delta), -abs(item.delta - theta)))

    def next_linear_item(self, administered: List[str], start_no: int = 1) -> Optional[ItemRecord]:
        used = set(administered)
        ordered = sorted(self.items, key=lambda x: x.no)
        if not ordered:
            return None
        start_no = int(start_no) if start_no is not None else 1
        start_no = max(1, min(start_no, len(ordered)))
        start_idx = 0
        for idx, item in enumerate(ordered):
            if item.no >= start_no:
                start_idx = idx
                break
        rotated = ordered[start_idx:] + ordered[:start_idx]
        for item in rotated:
            if item.item_id not in used:
                return item
        return None

    def sample_voice_items(self, center_theta: float, theta_range: float, n_items: int) -> List[ItemRecord]:
        n_items = max(1, int(n_items))
        theta_range = max(0.05, float(theta_range))
        eligible = [item for item in self.items if np.isfinite(item.delta) and abs(float(item.delta) - center_theta) <= theta_range]
        rng = random.SystemRandom()
        if len(eligible) > n_items:
            picked = rng.sample(eligible, n_items)
        else:
            picked = list(eligible)
        picked.sort(key=lambda item: (abs(float(item.delta) - center_theta), item.no))
        return picked

    def percentile(self, theta: float) -> float:
        vals = self.person_distribution
        if vals.size == 0:
            z = (theta - self.prior_mean) / self.prior_sd
            return 100.0 * (0.5 * (1 + math.erf(z / math.sqrt(2))))
        return 100.0 * float(np.mean(vals <= theta))

    def local_asset_path(self, raw_link: str) -> Optional[Path]:
        raw = (raw_link or "").strip().replace("\\", "/")
        if not raw or re.match(r"^[a-z]+://", raw, re.I):
            return None
        candidate = raw.lstrip("./")
        candidate_path = self.extract_dir / candidate
        if candidate_path.exists():
            return candidate_path
        local_path = Path(__file__).parent / candidate
        if local_path.exists():
            return local_path
        if "/" not in candidate and candidate.lower().endswith(IMG_EXTS):
            candidate = f"pic/{candidate}"
            candidate_path = self.extract_dir / candidate
            if candidate_path.exists():
                return candidate_path
            local_path = Path(__file__).parent / candidate
            if local_path.exists():
                return local_path
        return None


def ordered_options(options: Dict[str, str]) -> Dict[str, str]:
    if not options:
        return {}
    ordered: Dict[str, str] = {}
    for lab in OPTION_LABELS:
        if lab in options:
            ordered[lab] = options[lab]
    for lab in sorted(options.keys()):
        if lab not in ordered:
            ordered[lab] = options[lab]
    return ordered


def parse_item_text(text: str) -> Tuple[str, Dict[str, str]]:
    clean = re.sub(r"\s+", " ", text).strip()
    pattern_cn = re.compile(r"[（(]([A-E])[）)]")
    matches = list(pattern_cn.finditer(clean))
    if matches:
        stem = clean[: matches[0].start()].strip()
        options: Dict[str, str] = {}
        for i, match in enumerate(matches):
            label = match.group(1).upper()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(clean)
            options[label] = clean[start:end].strip(" ;；。 ")
        return stem, ordered_options(options)
    pattern_en = re.compile(r"(?:^|\s)([A-E])[\.、:：]\s*")
    matches = list(pattern_en.finditer(clean))
    if matches:
        stem = clean[: matches[0].start()].strip()
        options: Dict[str, str] = {}
        for i, match in enumerate(matches):
            label = match.group(1).upper()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(clean)
            options[label] = clean[start:end].strip(" ;；。 ")
        return stem, ordered_options(options)
    return clean, {}


def _svg_wrap(width: int, height: int, inner: str) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">{inner}</svg>'


def make_combined_kidmap_svg(person_values: np.ndarray, theta: float, person_se: float, rows: List[Dict[str, object]], infit_mnsq: float = 1.0, outfit_mnsq: float = 1.0) -> str:
    values = np.asarray(person_values, dtype=float)
    values = values[np.isfinite(values)]
    item_deltas = np.array([float(r.get("delta", 0.0)) for r in rows], dtype=float) if rows else np.array([], dtype=float)
    se_band = max(float(person_se or 0.0), 0.0)
    ymin = min(values.min() if values.size else theta, item_deltas.min() if item_deltas.size else theta, theta - se_band, theta) - 0.6
    ymax = max(values.max() if values.size else theta, item_deltas.max() if item_deltas.size else theta, theta + se_band, theta) + 0.6
    if ymax <= ymin:
        ymax = ymin + 1.0

    width, height = 980, 680
    left, right, top, bottom = 68, 28, 56, 56
    strip_w, gap_w = 152, 18
    x0, y0 = left, top
    plot_h = height - top - bottom
    resid_x0 = x0 + strip_w + gap_w
    plot_w = width - resid_x0 - right
    xmin, xmax = -4.0, 4.0

    def xmap(val: float) -> float:
        val = max(xmin, min(xmax, val))
        return resid_x0 + (val - xmin) / (xmax - xmin) * plot_w

    def ymap(val: float) -> float:
        return y0 + (ymax - val) / (ymax - ymin) * plot_h

    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>']

    title = f'KIDMAP — Measure {theta:.2f} (SE {se_band:.2f})  INFIT {infit_mnsq:.2f}  OUTFIT {outfit_mnsq:.2f}'
    parts.append(f'<text x="{x0}" y="26" font-size="16" font-weight="700" fill="#111827">{html.escape(title)}</text>')

    for t in range(math.floor(ymin), math.ceil(ymax) + 1):
        y = ymap(float(t))
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{resid_x0+plot_w}" y2="{y:.1f}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{x0-12}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#4b5563">{t}</text>')

    # left panel: person distribution as horizontal bars
    parts.append(f'<rect x="{x0}" y="{y0}" width="{strip_w}" height="{plot_h}" fill="#f8fafc" stroke="#e5e7eb"/>')
    parts.append(f'<text x="{x0 + strip_w/2:.1f}" y="46" text-anchor="middle" font-size="15" font-weight="700" fill="#1f2937">Persons (distribution)</text>')
    parts.append(f'<text x="{x0 + strip_w/2:.1f}" y="{height-16}" text-anchor="middle" font-size="11" fill="#64748b">Count</text>')

    if values.size:
        n_bins = max(14, min(28, int(round((ymax - ymin) * 4))))
        edges = np.linspace(ymin, ymax, n_bins + 1)
        counts, _ = np.histogram(values, bins=edges)
        max_count = max(int(counts.max()), 1)
        inner_left = x0 + 8
        usable_w = strip_w - 18
        for i, count in enumerate(counts):
            if count <= 0:
                continue
            y_top = ymap(edges[i + 1])
            y_bot = ymap(edges[i])
            bar_h = max(2.0, y_bot - y_top - 1.5)
            cy = y_top + (y_bot - y_top) / 2.0
            bar_w = usable_w * (count / max_count)
            parts.append(f'<rect x="{inner_left:.1f}" y="{cy - bar_h/2:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="2" fill="#4361c2" fill-opacity="0.82" stroke="#3148a5" stroke-opacity="0.55"/>')
        # simple count ticks
        for frac in (0.0, 0.5, 1.0):
            xv = inner_left + usable_w * frac
            label = int(round(max_count * frac))
            parts.append(f'<line x1="{xv:.1f}" y1="{y0+plot_h}" x2="{xv:.1f}" y2="{y0+plot_h+4}" stroke="#64748b"/>')
            parts.append(f'<text x="{xv:.1f}" y="{y0+plot_h+18}" text-anchor="middle" font-size="10" fill="#64748b">{label}</text>')

    sep_x = resid_x0 - gap_w/2
    parts.append(f'<line x1="{sep_x:.1f}" y1="{y0}" x2="{sep_x:.1f}" y2="{y0+plot_h}" stroke="#cbd5e1" stroke-width="1"/>')
    parts.append(f'<text x="{resid_x0 + plot_w/2:.1f}" y="46" text-anchor="middle" font-size="15" font-weight="700" fill="#1f2937">KIDMAP (cell ZSTD)</text>')

    # vertical ZSTD guides
    for zv in (-2, 2):
        x = xmap(float(zv))
        parts.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+plot_h}" stroke="#dc2626" stroke-width="2" stroke-dasharray="6,4"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-20}" text-anchor="middle" font-size="11" fill="#dc2626">{zv}</text>')
    x_zero = xmap(0.0)
    parts.append(f'<line x1="{x_zero:.1f}" y1="{y0}" x2="{x_zero:.1f}" y2="{y0+plot_h}" stroke="#cbd5e1" stroke-width="1.2" stroke-dasharray="4,4"/>')
    parts.append(f'<text x="{x_zero:.1f}" y="{height-20}" text-anchor="middle" font-size="11" fill="#64748b">0</text>')

    parts.append(f'<line x1="{resid_x0}" y1="{y0+plot_h}" x2="{resid_x0+plot_w}" y2="{y0+plot_h}" stroke="#374151"/>')
    parts.append(f'<text x="{(resid_x0+resid_x0+plot_w)/2:.1f}" y="{height-16}" text-anchor="middle" font-size="12">ZSTD</text>')
    parts.append(f'<text x="{x0-52}" y="{y0-8}" font-size="12" fill="#4b5563">Logit</text>')

    # person measure solid line and SE band dashed lines
    theta_y = ymap(theta)
    parts.append(f'<line x1="{x0}" y1="{theta_y:.1f}" x2="{resid_x0+plot_w}" y2="{theta_y:.1f}" stroke="#dc2626" stroke-width="2"/>')
    parts.append(f'<text x="{resid_x0+plot_w-6}" y="{max(16, theta_y-6):.1f}" text-anchor="end" font-size="12" fill="#991b1b">Measure {theta:.2f}</text>')
    if se_band > 1e-9:
        for band_val, band_lab in ((theta + se_band, '+SE'), (theta - se_band, '-SE')):
            by = ymap(band_val)
            parts.append(f'<line x1="{x0}" y1="{by:.1f}" x2="{resid_x0+plot_w}" y2="{by:.1f}" stroke="#dc2626" stroke-width="1.6" stroke-dasharray="4,4"/>')
            parts.append(f'<text x="{resid_x0+plot_w-6}" y="{max(16, by-4):.1f}" text-anchor="end" font-size="11" fill="#991b1b">{band_lab} ({band_val:.2f})</text>')

    if rows:
        ses = [float(r.get("item_se", 0.12) or 0.12) for r in rows]
        max_se = max(ses) if ses else 0.12
        min_se = min(ses) if ses else 0.12

        def rmap(se: float) -> float:
            if max_se <= min_se + 1e-9:
                return 8.0
            return 5.0 + (se - min_se) / (max_se - min_se) * 9.0

        for row in rows:
            delta = float(row.get("delta", 0.0))
            z = float(row.get("zscore", 0.0))
            se = float(row.get("item_se", 0.12) or 0.12)
            score = int(row.get("score", 0))
            item_id = html.escape(str(row.get("item_id", "")))
            x = xmap(z)
            y = ymap(delta)
            r = rmap(se)
            fill = '#2563eb' if score == 1 else '#dc2626'
            stroke = '#1e3a8a' if score == 1 else '#7f1d1d'
            stroke_w = 1.5 if abs(z) > 2 else 1.1
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" fill-opacity="0.78" stroke="{stroke}" stroke-width="{stroke_w}"/>')
            tx = min(resid_x0 + plot_w - 4, x + r + 5)
            parts.append(f'<text x="{tx:.1f}" y="{y+4:.1f}" font-size="10" fill="#111827">{item_id}</text>')

    legend_y = height - 38
    parts.append(f'<circle cx="{resid_x0+12}" cy="{legend_y}" r="6" fill="#2563eb" fill-opacity="0.78" stroke="#1e3a8a" stroke-width="1"/>')
    parts.append(f'<text x="{resid_x0+24}" y="{legend_y+4}" font-size="11">correct</text>')
    parts.append(f'<circle cx="{resid_x0+92}" cy="{legend_y}" r="6" fill="#dc2626" fill-opacity="0.78" stroke="#7f1d1d" stroke-width="1"/>')
    parts.append(f'<text x="{resid_x0+104}" y="{legend_y+4}" font-size="11">incorrect</text>')
    parts.append(f'<line x1="{resid_x0+186}" y1="{legend_y}" x2="{resid_x0+226}" y2="{legend_y}" stroke="#dc2626" stroke-width="2" stroke-dasharray="4,4"/>')
    parts.append(f'<text x="{resid_x0+234}" y="{legend_y+4}" font-size="11">Measure ± SE</text>')
    parts.append(f'<line x1="{resid_x0+330}" y1="{legend_y}" x2="{resid_x0+370}" y2="{legend_y}" stroke="#dc2626" stroke-width="2" stroke-dasharray="6,4"/>')
    parts.append(f'<text x="{resid_x0+378}" y="{legend_y+4}" font-size="11">ZSTD ±2</text>')
    parts.append(f'<text x="{resid_x0+462}" y="{legend_y+4}" font-size="11">bubble size ∝ item SE</text>')
    return _svg_wrap(width, height, ''.join(parts))


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    normalized = {re.sub(r'[^a-z0-9]+', '', str(col).lower()): col for col in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        key = re.sub(r'[^a-z0-9]+', '', cand.lower())
        if key in normalized:
            return normalized[key]
    return None


def make_home_wrightmap_svg(person_distribution: np.ndarray, item_df: pd.DataFrame) -> str:
    if item_df is None or item_df.empty:
        return _svg_wrap(980, 360, '<text x="20" y="40">No item statistics available for the homepage Wright Map.</text>')

    item_col = _find_column(item_df, ['ITEM', 'item'])
    measure_col = _find_column(item_df, ['MEASURE', 'measure'])
    infit_col = _find_column(item_df, ['INFIT_MNSQ', 'INFIT MNSQ', 'INFIT'])
    se_col = _find_column(item_df, ['SE', 'MODEL_SE', 'MODEL SE', 'S.E.', 'S.E'])

    if not item_col or not measure_col or not infit_col:
        return _svg_wrap(980, 360, '<text x="20" y="40">Missing ITEM / MEASURE / INFIT MNSQ columns for the homepage Wright Map.</text>')

    df = item_df.copy()
    df['item_id'] = df[item_col].astype(str)
    df['measure_val'] = pd.to_numeric(df[measure_col], errors='coerce')
    df['infit_val'] = pd.to_numeric(df[infit_col], errors='coerce')
    if se_col:
        df['se_val'] = pd.to_numeric(df[se_col], errors='coerce')
    else:
        df['se_val'] = 0.12
    df = df.dropna(subset=['measure_val', 'infit_val']).copy()
    if df.empty:
        return _svg_wrap(980, 360, '<text x="20" y="40">No finite MEASURE / INFIT MNSQ rows available for the homepage Wright Map.</text>')

    width, height = 980, 430
    left, right, top, bottom = 46, 28, 28, 70
    hist_w, gap = 150, 30
    x0, y0 = left, height - bottom
    resid_x0 = x0 + hist_w + gap
    plot_w = width - resid_x0 - right
    plot_h = height - top - bottom

    measures = df['measure_val'].astype(float).to_numpy()
    infits = df['infit_val'].astype(float).to_numpy()
    ses = df['se_val'].astype(float).fillna(0.12).to_numpy()

    person_vals = np.asarray(person_distribution if person_distribution is not None else np.array([]), dtype=float)
    person_vals = person_vals[np.isfinite(person_vals)]
    if person_vals.size == 0:
        person_vals = np.array([0.0])

    ymin = float(min(np.min(measures), np.min(person_vals)))
    ymax = float(max(np.max(measures), np.max(person_vals)))
    if ymax <= ymin:
        ymax = ymin + 1.0
    ypad = max(0.4, (ymax - ymin) * 0.08)
    ymin -= ypad
    ymax += ypad

    xmin = float(min(0.5, np.nanmin(infits)))
    xmax = float(max(2.0, np.nanmax(infits)))
    xpad = max(0.08, (xmax - xmin) * 0.05)
    xmin -= xpad
    xmax += xpad

    def ymap(v: float) -> float:
        return y0 - (v - ymin) / (ymax - ymin) * plot_h

    def xmap(v: float) -> float:
        return resid_x0 + (v - xmin) / (xmax - xmin) * plot_w

    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>']

    # person histogram
    bins = np.linspace(ymin, ymax, 22)
    counts, edges = np.histogram(person_vals, bins=bins)
    max_count = int(np.max(counts)) if len(counts) else 1
    max_count = max(max_count, 1)
    parts.append(f'<rect x="{x0}" y="{top}" width="{hist_w}" height="{plot_h}" fill="#f3f4f6" stroke="#d1d5db"/>')
    for c, y1, y2 in zip(counts, edges[:-1], edges[1:]):
        if c <= 0:
            continue
        yy1, yy2 = ymap(y1), ymap(y2)
        bh = max(1.0, abs(yy1 - yy2) - 1.0)
        bw = c / max_count * (hist_w - 18)
        by = min(yy1, yy2) + 0.5
        parts.append(f'<rect x="{x0}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="#5b7bd5" fill-opacity="0.9"/>')

    # axes and grid for item panel
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0 + hist_w}" y2="{y0}" stroke="#666"/>')
    parts.append(f'<line x1="{resid_x0}" y1="{y0}" x2="{resid_x0 + plot_w}" y2="{y0}" stroke="#666"/>')
    parts.append(f'<line x1="{resid_x0}" y1="{top}" x2="{resid_x0}" y2="{y0}" stroke="#666"/>')
    for frac in np.linspace(0.0, 1.0, 6):
        yv = ymin + (ymax - ymin) * frac
        yy = ymap(yv)
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{resid_x0 + plot_w}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x0 - 6}" y="{yy + 4:.1f}" text-anchor="end" font-size="11">{yv:.2f}</text>')

    # x ticks + boundary
    xticks = sorted(set([0.5, 1.0, 1.5, 2.0] + [round(v, 1) for v in np.linspace(max(0.0, xmin), xmax, 5)]))
    for xv in xticks:
        if xv < xmin or xv > xmax:
            continue
        xx = xmap(xv)
        dash = '4,4' if abs(xv - 1.5) < 1e-9 else 'none'
        stroke = '#dc2626' if abs(xv - 1.5) < 1e-9 else '#d1d5db'
        sw = '2' if abs(xv - 1.5) < 1e-9 else '1'
        parts.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{y0}" stroke="{stroke}" stroke-width="{sw}"' + (f' stroke-dasharray="{dash}"' if dash != 'none' else '') + '/>')
        parts.append(f'<text x="{xx:.1f}" y="{y0 + 18:.1f}" text-anchor="middle" font-size="11" fill="#111827">{xv:.1f}</text>')

    se_min = float(np.nanmin(ses)) if len(ses) else 0.12
    se_max = float(np.nanmax(ses)) if len(ses) else 0.12
    def rmap(se: float) -> float:
        if se_max <= se_min + 1e-9:
            return 7.0
        return 5.0 + (se - se_min) / (se_max - se_min) * 11.0

    for row in df.itertuples(index=False):
        item_id = html.escape(str(row.item_id))
        measure = float(row.measure_val)
        infit = float(row.infit_val)
        se = float(row.se_val) if np.isfinite(float(row.se_val)) else 0.12
        x = xmap(infit)
        y = ymap(measure)
        r = rmap(se)
        over = infit > 1.5
        fill = '#ef4444' if over else '#2563eb'
        stroke = '#991b1b' if over else '#1e3a8a'
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" fill-opacity="0.78" stroke="{stroke}" stroke-width="1.2">'
                     f'<title>{item_id} | Measure={measure:.3f} | INFIT MNSQ={infit:.3f} | SE={se:.3f}</title></circle>')
        if over:
            tx = min(resid_x0 + plot_w - 4, x + r + 4)
            parts.append(f'<text x="{tx:.1f}" y="{y + 4:.1f}" font-size="10" fill="#7f1d1d">{item_id}</text>')

    high_n = int(np.sum(infits > 1.5))
    parts.append(f'<text x="{x0 + hist_w/2:.1f}" y="16" text-anchor="middle" font-size="15" font-weight="700" fill="#1f2937">Persons (distribution)</text>')
    parts.append(f'<text x="{resid_x0 + plot_w/2:.1f}" y="16" text-anchor="middle" font-size="15" font-weight="700" fill="#1f2937">Items by INFIT MNSQ</text>')
    parts.append(f'<text x="{resid_x0 + plot_w/2:.1f}" y="{height - 18:.1f}" text-anchor="middle" font-size="12">INFIT MNSQ</text>')
    parts.append(f'<text x="16" y="{top + plot_h/2:.1f}" transform="rotate(-90 16 {top + plot_h/2:.1f})" text-anchor="middle" font-size="12">Item / person measure</text>')

    legend_y = height - 42
    parts.append(f'<circle cx="{resid_x0 + 12:.1f}" cy="{legend_y:.1f}" r="6" fill="#2563eb" fill-opacity="0.78" stroke="#1e3a8a" stroke-width="1"/>')
    parts.append(f'<text x="{resid_x0 + 24:.1f}" y="{legend_y + 4:.1f}" font-size="11">INFIT ≤ 1.5</text>')
    parts.append(f'<circle cx="{resid_x0 + 118:.1f}" cy="{legend_y:.1f}" r="6" fill="#ef4444" fill-opacity="0.78" stroke="#991b1b" stroke-width="1"/>')
    parts.append(f'<text x="{resid_x0 + 130:.1f}" y="{legend_y + 4:.1f}" font-size="11">INFIT > 1.5</text>')
    parts.append(f'<line x1="{resid_x0 + 220:.1f}" y1="{legend_y:.1f}" x2="{resid_x0 + 260:.1f}" y2="{legend_y:.1f}" stroke="#dc2626" stroke-width="2" stroke-dasharray="4,4"/>')
    parts.append(f'<text x="{resid_x0 + 268:.1f}" y="{legend_y + 4:.1f}" font-size="11">Boundary 1.5</text>')
    parts.append(f'<text x="{resid_x0 + 382:.1f}" y="{legend_y + 4:.1f}" font-size="11">bubble size ∝ item SE</text>')
    parts.append(f'<text x="{resid_x0 + plot_w - 4:.1f}" y="{legend_y + 4:.1f}" text-anchor="end" font-size="11" fill="#991b1b">Items beyond 1.5: {high_n}</text>')
    return _svg_wrap(width, height, ''.join(parts))


def make_itemfit_svg(item_df: pd.DataFrame, history: List[dict]) -> str:
    if item_df is None or item_df.empty or 'MEASURE' not in item_df.columns or 'ITEM' not in item_df.columns:
        return _svg_wrap(900, 260, '<text x="20" y="40">No item fit Z-score data available.</text>')
    df = item_df.copy()
    df['ITEM'] = df['ITEM'].astype(str)
    xvals = pd.to_numeric(df['MEASURE'], errors='coerce')
    bank_yvals = pd.to_numeric(df.get('INFIT_ZSTD'), errors='coerce') if 'INFIT_ZSTD' in df.columns else pd.Series(np.nan, index=df.index)
    keep = xvals.notna()
    df = df.loc[keep, ['ITEM']].copy()
    df['x'] = xvals[keep].astype(float).to_numpy()
    df['bank_y'] = bank_yvals[keep].astype(float).to_numpy() if len(bank_yvals) else np.full(len(df), np.nan)

    resp_rows = []
    for row in history or []:
        item_id = str(row.get('item_id', ''))
        if not item_id:
            continue
        item = BANK.item_lookup.get(item_id)
        if item is None:
            continue
        try:
            z = float(row.get('zstd', np.nan))
        except Exception:
            z = float('nan')
        if not np.isfinite(z):
            continue
        resp_rows.append({'ITEM': item_id, 'x': float(item.delta), 'resp_y': z})
    resp_df = pd.DataFrame(resp_rows)

    y_candidates = [v for v in pd.to_numeric(df['bank_y'], errors='coerce').tolist() if np.isfinite(v)]
    if not resp_df.empty:
        y_candidates.extend([v for v in pd.to_numeric(resp_df['resp_y'], errors='coerce').tolist() if np.isfinite(v)])
    if not y_candidates:
        y_candidates = [-3.0, 3.0]

    width, height = 900, 290
    left, right, top, bottom = 52, 18, 20, 64
    x0, y0 = left, height - bottom
    plot_w, plot_h = width - left - right, height - top - bottom
    xmin, xmax = float(df['x'].min()), float(df['x'].max())
    ymin, ymax = float(min(-3.0, min(y_candidates) - 0.5)), float(max(3.0, max(y_candidates) + 0.5))
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
             f'<line x1="{x0}" y1="{y0}" x2="{x0+plot_w}" y2="{y0}" stroke="#444"/>',
             f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{y0}" stroke="#444"/>']
    for yv in [-2, 0, 2]:
        yy = y0 - (yv - ymin) / (ymax - ymin) * plot_h
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+plot_w}" y2="{yy:.1f}" stroke="#d1d5db" stroke-dasharray="4,4"/>')
        parts.append(f'<text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11">{yv}</text>')

    for row in df.itertuples(index=False):
        if not np.isfinite(float(row.bank_y)):
            continue
        x = x0 + (float(row.x) - xmin) / (xmax - xmin) * plot_w
        y = y0 - (float(row.bank_y) - ymin) / (ymax - ymin) * plot_h
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="#6b7280" opacity="0.65"/>')

    if not resp_df.empty:
        for row in resp_df.itertuples(index=False):
            x = x0 + (float(row.x) - xmin) / (xmax - xmin) * plot_w
            y = y0 - (float(row.resp_y) - ymin) / (ymax - ymin) * plot_h
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#dc2626" opacity="0.9"/>')

    legend_x = x0 + 8
    legend_y = y0 + 28
    parts.append(f'<circle cx="{legend_x:.1f}" cy="{legend_y:.1f}" r="4.5" fill="#dc2626" opacity="0.9"/>')
    parts.append(f'<text x="{legend_x + 12:.1f}" y="{legend_y + 4:.1f}" font-size="11">Red dots = responded items (response ZSTD)</text>')
    parts.append(f'<circle cx="{legend_x + 280:.1f}" cy="{legend_y:.1f}" r="2.8" fill="#6b7280" opacity="0.65"/>')
    parts.append(f'<text x="{legend_x + 292:.1f}" y="{legend_y + 4:.1f}" font-size="11">Gray dots = bank items (INFIT ZSTD)</text>')
    parts.append(f'<text x="{x0 + plot_w/2:.1f}" y="{height - 14}" text-anchor="middle" font-size="12">Item difficulty</text>')
    parts.append(f'<text x="18" y="{top + plot_h/2:.1f}" transform="rotate(-90 18 {top + plot_h/2:.1f})" text-anchor="middle" font-size="12">ZSTD</text>')
    return _svg_wrap(width, height, ''.join(parts))


def compute_person_fit(responses: List[Tuple[str, int]], theta: float) -> Tuple[float, float]:
    if not responses:
        return 1.0, 1.0
    numer = 0.0
    denom = 0.0
    outfit_terms = []
    for item_id, score in responses:
        item = BANK.item_lookup.get(item_id)
        if not item:
            continue
        p = float(np.clip(BANK.probability(theta, item.delta), 1e-8, 1 - 1e-8))
        var = max(p * (1.0 - p), 1e-8)
        resid2 = (float(score) - p) ** 2
        numer += resid2
        denom += var
        outfit_terms.append(resid2 / var)
    infit = numer / denom if denom > 0 else 1.0
    outfit = float(np.mean(outfit_terms)) if outfit_terms else 1.0
    return infit, outfit


def build_dashboard_data(state: dict) -> Dict[str, object]:
    responses = [tuple(x) for x in state.get('responses', [])]
    history = state.get('history', [])
    answered_ids = [item_id for item_id, _ in responses]
    final_theta = float(state.get('theta', BANK.prior_mean))
    residual_rows = []
    score_lookup = {item_id: int(score) for item_id, score in responses}
    item_stats = BANK.item_fit_df.copy() if isinstance(BANK.item_fit_df, pd.DataFrame) else pd.DataFrame()
    item_se_map = {}
    if not item_stats.empty and 'ITEM' in item_stats.columns and 'SE' in item_stats.columns:
        item_se_map = dict(zip(item_stats['ITEM'].astype(str), pd.to_numeric(item_stats['SE'], errors='coerce').fillna(0.12)))
    for row in history:
        item_id = str(row.get('item_id', ''))
        item = BANK.item_lookup.get(item_id)
        if not item:
            continue
        score = int(score_lookup.get(item_id, 0))
        p = float(np.clip(BANK.probability(final_theta, item.delta), 1e-8, 1 - 1e-8))
        z = (score - p) / math.sqrt(max(p * (1 - p), 1e-8))
        residual_rows.append({
            'item_id': item_id,
            'delta': float(item.delta),
            'score': score,
            'zscore': float(z),
            'item_se': float(item_se_map.get(item_id, 0.12) or 0.12),
        })
    infit_mnsq, outfit_mnsq = compute_person_fit(responses, final_theta)
    final_se = float(state.get('se', BANK.prior_sd))
    cpc_svg, cpc_pred_label = make_cpc_svg(final_theta, ref_delta=0.0)
    return {
        'answered_ids': answered_ids,
        'kidmap_svg': make_combined_kidmap_svg(BANK.person_distribution, final_theta, final_se, residual_rows, infit_mnsq, outfit_mnsq),
        'itemfit_svg': make_itemfit_svg(BANK.item_fit_df, history),
        'infit_mnsq': infit_mnsq,
        'outfit_mnsq': outfit_mnsq,
        'trend_svg': make_trend_svg(history),
        'cpc_svg': cpc_svg,
        'cpc_pred_label': cpc_pred_label,
        'comparison': build_cat_noncat_comparison(state),
        'comparison_n': build_cat_noncat_n_comparison(state),
    }


def make_cpc_svg(theta_person: float, ref_delta: float = 0.0) -> Tuple[str, str]:
    grid = np.linspace(-6.0, 6.0, 901)
    p1 = np.asarray([BANK.probability(float(t), float(ref_delta)) for t in grid], dtype=float)
    p1 = np.clip(p1, 1e-9, 1 - 1e-9)
    p0 = 1.0 - p1
    width, height = 980, 420
    left, right, top, bottom = 62, 24, 28, 52
    x0, y0 = left, height - bottom
    plot_w, plot_h = width - left - right, height - top - bottom
    xmin, xmax = float(grid.min()), float(grid.max())
    def xmap(v: float) -> float:
        return x0 + (v - xmin) / (xmax - xmin) * plot_w
    def ymap(v: float) -> float:
        return y0 - v * plot_h
    parts = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0+plot_w}" y2="{y0}" stroke="#444"/>',
        f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{y0}" stroke="#444"/>',
    ]
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        yy = ymap(frac)
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+plot_w}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11">{frac:.2f}</text>')
    for xv in range(-6, 7):
        xx = xmap(float(xv))
        parts.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0+4}" stroke="#64748b"/>')
        parts.append(f'<text x="{xx:.1f}" y="{y0+18:.1f}" text-anchor="middle" font-size="11">{xv}</text>')
    pts0 = [f"{xmap(grid[i]):.1f},{ymap(float(p0[i])):.1f}" for i in range(len(grid))]
    pts1 = [f"{xmap(grid[i]):.1f},{ymap(float(p1[i])):.1f}" for i in range(len(grid))]
    parts.append(f'<path d="M ' + ' L '.join(pts0) + '" fill="none" stroke="#2563eb" stroke-width="2.3"/>')
    parts.append(f'<path d="M ' + ' L '.join(pts1) + '" fill="none" stroke="#dc2626" stroke-width="2.3"/>')
    # intersection for dichotomous Rasch with delta = 0
    xx0 = xmap(float(ref_delta))
    parts.append(f'<line x1="{xx0:.1f}" y1="{top}" x2="{xx0:.1f}" y2="{y0}" stroke="#dc2626" stroke-width="1.8" stroke-dasharray="4,4"/>')
    parts.append(f'<text x="{xx0:.1f}" y="{top-8 if top>10 else 12}" text-anchor="middle" font-size="10" fill="#dc2626">delta = {ref_delta:.2f}</text>')
    xxp = xmap(float(theta_person))
    parts.append(f'<line x1="{xxp:.1f}" y1="{top}" x2="{xxp:.1f}" y2="{y0}" stroke="#b91c1c" stroke-width="2.4"/>')
    parts.append(f'<text x="{xxp:.1f}" y="{y0+34:.1f}" text-anchor="middle" font-size="11" fill="#b91c1c">Person θ = {theta_person:.2f}</text>')
    p1_person = float(np.clip(BANK.probability(float(theta_person), float(ref_delta)), 1e-9, 1 - 1e-9))
    p0_person = 1.0 - p1_person
    y1p = ymap(p1_person)
    y0p = ymap(p0_person)
    parts.append(f'<circle cx="{xxp:.1f}" cy="{y1p:.1f}" r="4.3" fill="#dc2626" stroke="white" stroke-width="1"/>')
    parts.append(f'<circle cx="{xxp:.1f}" cy="{y0p:.1f}" r="4.3" fill="#2563eb" stroke="white" stroke-width="1"/>')
    parts.append(f'<text x="{min(x0+plot_w-6, xxp+8):.1f}" y="{y1p-8:.1f}" font-size="11" fill="#dc2626">P(1)={p1_person:.2f}</text>')
    parts.append(f'<text x="{min(x0+plot_w-6, xxp+8):.1f}" y="{y0p+14:.1f}" font-size="11" fill="#2563eb">P(0)={p0_person:.2f}</text>')
    leg_x, leg_y = x0 + 12, top + 12
    parts.append(f'<line x1="{leg_x}" y1="{leg_y}" x2="{leg_x+22}" y2="{leg_y}" stroke="#2563eb" stroke-width="2.4"/>')
    parts.append(f'<text x="{leg_x+28}" y="{leg_y+4}" font-size="11">Incorrect (0)</text>')
    parts.append(f'<line x1="{leg_x}" y1="{leg_y+18}" x2="{leg_x+22}" y2="{leg_y+18}" stroke="#dc2626" stroke-width="2.4"/>')
    parts.append(f'<text x="{leg_x+28}" y="{leg_y+22}" font-size="11">Correct (1)</text>')
    parts.append(f'<text x="{x0 + plot_w/2:.1f}" y="{height - 12:.1f}" text-anchor="middle" font-size="12">Theta for reference item (delta = 0)</text>')
    parts.append(f'<text x="18" y="{top + plot_h/2:.1f}" transform="rotate(-90 18 {top + plot_h/2:.1f})" text-anchor="middle" font-size="12">Category probability</text>')
    pred_label = 'Correct (1)' if p1_person >= 0.5 else 'Incorrect (0)'
    return _svg_wrap(width, height, ''.join(parts)), pred_label


def onesample_ttest_against_constant(values: List[float], mu: float) -> dict:
    arr = np.asarray([float(x) for x in values if np.isfinite(float(x))], dtype=float)
    if arr.size < 2:
        mean = float(np.mean(arr)) if arr.size else 0.0
        return {'n': int(arr.size), 'mean': mean, 'sd': 0.0, 't': 0.0, 'p': None, 'df': None, 'p_text': 'NA', 'df_text': 'NA', 't_text': 'NA', 'note': 'Too few simulated CAT runs for a t-test.'}
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    if sd < 1e-12:
        note = f'All CAT runs stopped at {mean:.3f} items, so the CAT item length had zero variance and the t-test is not informative.'
        return {'n': int(arr.size), 'mean': mean, 'sd': sd, 't': 0.0, 'p': None, 'df': int(arr.size - 1), 'p_text': 'NA', 'df_text': str(arr.size - 1), 't_text': 'NA', 'note': note}
    se = sd / math.sqrt(arr.size)
    t = (mean - float(mu)) / se
    try:
        from scipy import stats
        p = float(2.0 * stats.t.sf(abs(t), df=arr.size - 1))
    except Exception:
        p = float(math.erfc(abs(t) / math.sqrt(2.0)))
    return {'n': int(arr.size), 'mean': mean, 'sd': sd, 't': float(t), 'p': p, 'df': int(arr.size - 1), 'p_text': f'{p:.4g}', 'df_text': str(arr.size - 1), 't_text': f'{t:.3f}', 'note': ''}


def paired_ttest(a: List[float], b: List[float]) -> dict:
    n = min(len(a), len(b))
    if n <= 0:
        return {'n': 0, 'mean': 0.0, 'sd': 0.0, 't': 0.0, 'p': None, 'df': 0, 'p_text': 'NA', 'df_text': 'NA', 't_text': 'NA'}
    diffs = [float(a[i]) - float(b[i]) for i in range(n)]
    out = onesample_ttest_against_constant(diffs, 0.0)
    out['mean'] = float(np.mean(diffs)) if diffs else 0.0
    out['sd'] = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    return out


def summarize_mean_sd(values: List[float]) -> dict:
    arr = np.asarray([float(x) for x in values if np.isfinite(float(x))], dtype=float)
    if arr.size == 0:
        return {'n': 0, 'mean': 0.0, 'sd': 0.0}
    return {'n': int(arr.size), 'mean': float(np.mean(arr)), 'sd': float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0}


def _box_stats(values: List[float]) -> Optional[dict]:
    arr = np.asarray([float(x) for x in values if np.isfinite(float(x))], dtype=float)
    if arr.size == 0:
        return None
    return {'min': float(np.min(arr)), 'q1': float(np.percentile(arr, 25)), 'med': float(np.percentile(arr, 50)), 'q3': float(np.percentile(arr, 75)), 'max': float(np.max(arr)), 'mean': float(np.mean(arr))}


def _draw_boxplot(parts: List[str], x: float, stats: dict, ymap, color: str, stroke: str, box_w: float = 70.0, show_points: Optional[List[float]] = None):
    q1, med, q3 = stats['q1'], stats['med'], stats['q3']
    vmin, vmax = stats['min'], stats['max']
    parts.append(f'<line x1="{x:.1f}" y1="{ymap(vmin):.1f}" x2="{x:.1f}" y2="{ymap(vmax):.1f}" stroke="{stroke}" stroke-width="1.8"/>')
    parts.append(f'<line x1="{x-box_w/4:.1f}" y1="{ymap(vmin):.1f}" x2="{x+box_w/4:.1f}" y2="{ymap(vmin):.1f}" stroke="{stroke}" stroke-width="1.8"/>')
    parts.append(f'<line x1="{x-box_w/4:.1f}" y1="{ymap(vmax):.1f}" x2="{x+box_w/4:.1f}" y2="{ymap(vmax):.1f}" stroke="{stroke}" stroke-width="1.8"/>')
    parts.append(f'<rect x="{x-box_w/2:.1f}" y="{ymap(q3):.1f}" width="{box_w:.1f}" height="{max(1.2, ymap(q1)-ymap(q3)):.1f}" fill="{color}" fill-opacity="0.55" stroke="{stroke}"/>')
    parts.append(f'<line x1="{x-box_w/2:.1f}" y1="{ymap(med):.1f}" x2="{x+box_w/2:.1f}" y2="{ymap(med):.1f}" stroke="{stroke}" stroke-width="2.5"/>')
    parts.append(f'<circle cx="{x:.1f}" cy="{ymap(stats["mean"]):.1f}" r="4.2" fill="#2563eb" stroke="white" stroke-width="1"/>')
    if show_points:
        rng = random.Random(42 + int(x))
        for v in show_points:
            xx = x + rng.uniform(-box_w*0.34, box_w*0.34)
            parts.append(f'<circle cx="{xx:.1f}" cy="{ymap(float(v)):.1f}" r="2.9" fill="{stroke}" fill-opacity="0.55"/>')


def make_length_efficiency_svg(full_length: int, cat_lengths: List[float]) -> str:
    width, height = 920, 320
    left, right, top, bottom = 58, 24, 32, 46
    plot_w, plot_h = width - left - right, height - top - bottom
    all_vals = [float(full_length)] + [float(v) for v in cat_lengths] if cat_lengths else [float(full_length)]
    y_min = min(all_vals) - 1.0
    y_max = max(all_vals) + 1.0
    if y_max <= y_min:
        y_max = y_min + 2.0
    def ymap(v: float) -> float:
        return top + plot_h - (v - y_min) / (y_max - y_min) * plot_h
    x_nat = left + plot_w * 0.28
    x_cat = left + plot_w * 0.72
    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>']
    for yv in np.linspace(math.floor(y_min), math.ceil(y_max), 6):
        yy = ymap(yv)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11">{yv:.0f}</text>')
    full_stats = {'min': float(full_length), 'q1': float(full_length), 'med': float(full_length), 'q3': float(full_length), 'max': float(full_length), 'mean': float(full_length)}
    _draw_boxplot(parts, x_nat, full_stats, ymap, '#f59e0b', '#b45309', box_w=72, show_points=[float(full_length)])
    if cat_lengths:
        _draw_boxplot(parts, x_cat, _box_stats(cat_lengths), ymap, '#fca5a5', '#b91c1c', box_w=78, show_points=cat_lengths)
    parts.append(f'<text x="{left+plot_w/2:.1f}" y="18" text-anchor="middle" font-size="15" font-weight="700">a  Item length</text>')
    parts.append(f'<text x="{x_nat:.1f}" y="{height-16:.1f}" text-anchor="middle" font-size="12">Full non-CAT</text>')
    parts.append(f'<text x="{x_cat:.1f}" y="{height-16:.1f}" text-anchor="middle" font-size="12">CAT</text>')
    parts.append(f'<text x="18" y="{top+plot_h/2:.1f}" transform="rotate(-90 18 {top+plot_h/2:.1f})" text-anchor="middle" font-size="12">Item length</text>')
    return _svg_wrap(width, height, ''.join(parts))


def make_theta_boxplot_svg(full_thetas: List[float], cat_thetas: List[float]) -> str:
    width, height = 920, 320
    left, right, top, bottom = 58, 24, 32, 46
    plot_w, plot_h = width - left - right, height - top - bottom
    all_vals = [float(v) for v in (full_thetas + cat_thetas) if np.isfinite(float(v))]
    if not all_vals:
        all_vals = [0.0]
    y_min = min(all_vals) - 0.3
    y_max = max(all_vals) + 0.3
    if y_max <= y_min:
        y_max = y_min + 1.0
    def ymap(v: float) -> float:
        return top + plot_h - (v - y_min) / (y_max - y_min) * plot_h
    x_nat = left + plot_w * 0.28
    x_cat = left + plot_w * 0.72
    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>']
    for yv in np.linspace(y_min, y_max, 6):
        yy = ymap(yv)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11">{yv:.1f}</text>')
    if full_thetas:
        _draw_boxplot(parts, x_nat, _box_stats(full_thetas), ymap, '#86efac', '#15803d', box_w=72, show_points=full_thetas)
    if cat_thetas:
        _draw_boxplot(parts, x_cat, _box_stats(cat_thetas), ymap, '#fecaca', '#b91c1c', box_w=78, show_points=cat_thetas)
    parts.append(f'<text x="{left+plot_w/2:.1f}" y="18" text-anchor="middle" font-size="15" font-weight="700">b  Person measure</text>')
    parts.append(f'<text x="{x_nat:.1f}" y="{height-16:.1f}" text-anchor="middle" font-size="12">Full non-CAT</text>')
    parts.append(f'<text x="{x_cat:.1f}" y="{height-16:.1f}" text-anchor="middle" font-size="12">CAT</text>')
    parts.append(f'<text x="18" y="{top+plot_h/2:.1f}" transform="rotate(-90 18 {top+plot_h/2:.1f})" text-anchor="middle" font-size="12">Person measure</text>')
    return _svg_wrap(width, height, ''.join(parts))


def _make_fixed_score_map(target_correct_prop: float = 0.75, rng: Optional[random.Random] = None) -> Dict[str, int]:
    rng = rng or random.Random()
    item_ids = [it.item_id for it in BANK.items]
    n_items = len(item_ids)
    n_correct = int(round(float(target_correct_prop) * n_items))
    n_correct = max(0, min(n_correct, n_items))
    correct_ids = set(rng.sample(item_ids, n_correct))
    return {item_id: (1 if item_id in correct_ids else 0) for item_id in item_ids}


def _pick_answer_for_score(item: ItemRecord, score: int, rng: random.Random) -> str:
    opts = list((item.options_zh or item.options_en or {}).keys())
    if not opts:
        opts = OPTION_LABELS[:4]
    if int(score) == 1:
        return str(item.key).upper()
    wrong = [str(x).upper() for x in opts if str(x).upper() != str(item.key).upper()]
    return str(rng.choice(wrong if wrong else opts)).upper()




def make_length_boxplot_svg(full_lengths: List[float], cat_lengths: List[float]) -> str:
    width, height = 920, 320
    left, right, top, bottom = 58, 24, 32, 46
    plot_w, plot_h = width - left - right, height - top - bottom
    all_vals = [float(v) for v in (full_lengths + cat_lengths) if np.isfinite(float(v))]
    if not all_vals:
        all_vals = [0.0]
    y_min = min(all_vals) - 1.0
    y_max = max(all_vals) + 1.0
    if y_max <= y_min:
        y_max = y_min + 2.0
    def ymap(v: float) -> float:
        return top + plot_h - (v - y_min) / (y_max - y_min) * plot_h
    x_nat = left + plot_w * 0.28
    x_cat = left + plot_w * 0.72
    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>']
    for yv in np.linspace(math.floor(y_min), math.ceil(y_max), 6):
        yy = ymap(yv)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11">{yv:.0f}</text>')
    if full_lengths:
        _draw_boxplot(parts, x_nat, _box_stats(full_lengths), ymap, '#fde68a', '#a16207', box_w=72, show_points=full_lengths)
    if cat_lengths:
        _draw_boxplot(parts, x_cat, _box_stats(cat_lengths), ymap, '#93c5fd', '#1d4ed8', box_w=72, show_points=cat_lengths)
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#444"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#444"/>')
    parts.append(f'<text x="{x_nat:.1f}" y="{top+plot_h+24}" text-anchor="middle" font-size="12">Full non-CAT</text>')
    parts.append(f'<text x="{x_cat:.1f}" y="{top+plot_h+24}" text-anchor="middle" font-size="12">CAT</text>')
    parts.append(f'<text x="{width/2:.1f}" y="{height-8}" text-anchor="middle" font-size="12">Item length</text>')
    parts.append(f'<text x="14" y="{height/2:.1f}" text-anchor="middle" font-size="12" transform="rotate(-90 14 {height/2:.1f})">Item length</text>')
    return _svg_wrap(width, height, ''.join(parts))
def _run_full_noncat_with_fixed_map(score_map: Dict[str, int], start_theta: float) -> dict:
    responses: List[Tuple[str, int]] = []
    for item in BANK.items:
        if item.item_id in score_map:
            responses.append((item.item_id, int(score_map.get(item.item_id, 0))))
    theta, se, _ = BANK.posterior(responses, start_theta=start_theta)
    return {'theta': float(theta), 'se': float(se), 'length': len(responses), 'responses': responses}


def _build_score_map_from_original_row(row: pd.Series) -> Dict[str, int]:
    score_map: Dict[str, int] = {}
    for item_id in getattr(BANK, 'original_response_item_ids', []):
        val = row.get(item_id)
        if pd.isna(val):
            continue
        try:
            score = int(round(float(val)))
        except Exception:
            continue
        if score in (0, 1):
            score_map[str(item_id)] = int(score)
    return score_map


def _sample_original_person_score_map(rng: random.Random) -> dict:
    df = getattr(BANK, 'original_response_df', pd.DataFrame())
    if df is None or df.empty or not getattr(BANK, 'original_response_item_ids', []):
        score_map = _make_fixed_score_map(0.75, rng)
        return {'person_id': 'synthetic_75pct', 'row_index': -1, 'score_map': score_map}
    row_index = int(rng.randrange(len(df)))
    row = df.iloc[row_index]
    person_col = getattr(BANK, 'original_response_person_col', 'person_id')
    person_id = str(row.get(person_col, f'row_{row_index}'))
    score_map = _build_score_map_from_original_row(row)
    if not score_map:
        score_map = _make_fixed_score_map(0.75, rng)
    return {'person_id': person_id, 'row_index': row_index, 'score_map': score_map}


def _score_map_from_compare_state(state: dict, rng: Optional[random.Random] = None) -> dict:
    rng = rng or random.Random()
    row_index = state.get('compare_person_index', None)
    df = getattr(BANK, 'original_response_df', pd.DataFrame())
    if isinstance(row_index, int) and row_index >= 0 and row_index < len(df):
        row = df.iloc[int(row_index)]
        score_map = _build_score_map_from_original_row(row)
        person_col = getattr(BANK, 'original_response_person_col', 'person_id')
        person_id = str(row.get(person_col, state.get('compare_person_id', f'row_{row_index}')))
        return {'person_id': person_id, 'row_index': int(row_index), 'score_map': score_map}
    return _sample_original_person_score_map(rng)


def _sample_n_original_person_infos(rng: random.Random, n: int) -> List[dict]:
    df = getattr(BANK, 'original_response_df', pd.DataFrame())
    if df is None or df.empty or not getattr(BANK, 'original_response_item_ids', []):
        out = []
        for i in range(max(1, int(n))):
            score_map = _make_fixed_score_map(0.75, rng)
            out.append({'person_id': f'synthetic_{i+1:03d}', 'row_index': -1, 'score_map': score_map})
        return out
    total = len(df)
    n = max(1, min(int(n), total))
    indices = rng.sample(range(total), n)
    person_col = getattr(BANK, 'original_response_person_col', 'person_id')
    out: List[dict] = []
    for row_index in indices:
        row = df.iloc[int(row_index)]
        score_map = _build_score_map_from_original_row(row)
        if not score_map:
            continue
        person_id = str(row.get(person_col, f'row_{row_index}'))
        out.append({'person_id': person_id, 'row_index': int(row_index), 'score_map': score_map})
    return out


def _person_infos_from_compare_n_state(state: dict, rng: Optional[random.Random] = None) -> List[dict]:
    rng = rng or random.Random()
    df = getattr(BANK, 'original_response_df', pd.DataFrame())
    person_col = getattr(BANK, 'original_response_person_col', 'person_id')
    idxs = state.get('compare_n_person_indices', []) or []
    ids = state.get('compare_n_person_ids', []) or []
    out: List[dict] = []
    if isinstance(idxs, list) and len(idxs) > 0 and df is not None and not df.empty:
        for j, row_index in enumerate(idxs):
            try:
                row_index = int(row_index)
            except Exception:
                continue
            if 0 <= row_index < len(df):
                row = df.iloc[row_index]
                score_map = _build_score_map_from_original_row(row)
                if not score_map:
                    continue
                fallback_id = ids[j] if j < len(ids) else f'row_{row_index}'
                person_id = str(row.get(person_col, fallback_id))
                out.append({'person_id': person_id, 'row_index': row_index, 'score_map': score_map})
    if out:
        return out
    n = int(state.get('compare_n_persons', state.get('compare_n_reps', 20)) or 20)
    return _sample_n_original_person_infos(rng, n)


def _select_next_item_from_score_map(used: List[str], theta: float, score_map: Dict[str, int], rng: random.Random, random_first: bool = False) -> Optional[ItemRecord]:
    used_set = set(str(x) for x in used)
    remaining = [item for item in BANK.items if item.item_id not in used_set and item.item_id in score_map]
    if not remaining:
        return None
    if random_first and not used:
        return rng.choice(remaining)
    return max(remaining, key=lambda item: (BANK.information(theta, item.delta), -abs(item.delta - theta), -item.no))


def _run_cat_with_fixed_map(score_map: Dict[str, int], start_theta: float, stop_se: float, safety_cap: int, rng: random.Random, random_first: bool = False) -> dict:
    responses: List[Tuple[str, int]] = []
    history: List[dict] = []
    theta, se, _ = BANK.posterior([], start_theta=start_theta)
    available_item_count = max(1, len([item for item in BANK.items if item.item_id in score_map]))
    safety_cap = max(1, min(int(safety_cap), available_item_count))
    while len(responses) < safety_cap:
        used = [item_id for item_id, _ in responses]
        item = _select_next_item_from_score_map(used, theta, score_map, rng, random_first=random_first)
        if item is None:
            break
        score = int(score_map.get(item.item_id, 0))
        answer = _pick_answer_for_score(item, score, rng)
        theta_before = float(theta)
        p_before = float(np.clip(BANK.probability(theta_before, item.delta), 1e-8, 1 - 1e-8))
        zstd = (score - p_before) / math.sqrt(max(p_before * (1.0 - p_before), 1e-8))
        responses.append((item.item_id, score))
        theta, se, _ = BANK.posterior(responses, start_theta=start_theta)
        link_href, _ = resolve_link_href(item.link)
        history.append({'item_id': item.item_id, 'no': item.no, 'delta': float(item.delta), 'answer': answer, 'correct_tf': 'TRUE' if score == 1 else 'FALSE', 'theta_before': theta_before, 'theta': float(theta), 'se': float(se), 'zstd': float(zstd), 'link_href': link_href})
        if se <= float(stop_se):
            return {'theta': float(theta), 'se': float(se), 'length': len(responses), 'responses': responses, 'history': history, 'stop_reason': 'target_se'}
    return {'theta': float(theta), 'se': float(se), 'length': len(responses), 'responses': responses, 'history': history, 'stop_reason': 'all_items' if len(responses) >= available_item_count else 'max_items'}


def build_cat_noncat_comparison(state: dict) -> Optional[dict]:
    if str(state.get('mode', 'cat')) != 'compare':
        return None
    n_reps = int(state.get('compare_n_reps', 20) or 20)
    n_reps = max(2, n_reps)
    start_theta = float(state.get('start_theta', BANK.prior_mean))
    stop_se = float(state.get('stop_se', 0.32) or 0.32)
    person_info = _score_map_from_compare_state(state, random.Random(20260328))
    score_map = dict(person_info.get('score_map', {}) or {})
    if not score_map:
        return None
    full_run = _run_full_noncat_with_fixed_map(score_map, start_theta)
    full_length = int(full_run['length'])
    if full_length <= 0:
        return None
    full_theta = float(full_run['theta'])
    rng = random.Random(20260328 + max(0, int(person_info.get('row_index', 0))))
    cat_lengths: List[float] = []
    cat_thetas: List[float] = []
    for _ in range(n_reps):
        cat_run = _run_cat_with_fixed_map(score_map, start_theta, stop_se, full_length, rng, random_first=True)
        cat_lengths.append(float(cat_run['length']))
        cat_thetas.append(float(cat_run['theta']))
    history = list(state.get('history', []))
    actual_rows = [{'pos': idx, 'item_id': row.get('item_id', ''), 'no': row.get('no', ''), 'delta': float(row.get('delta', 0.0)), 'link_href': row.get('link_href', '')} for idx, row in enumerate(history, start=1)]
    full_length_stats = {'min': float(full_length), 'q1': float(full_length), 'med': float(full_length), 'q3': float(full_length), 'max': float(full_length), 'mean': float(full_length)}
    full_theta_single = [full_theta]
    cat_length_stats = _box_stats(cat_lengths) or full_length_stats
    full_theta_stats = _box_stats(full_theta_single) or {'min': full_theta, 'q1': full_theta, 'med': full_theta, 'q3': full_theta, 'max': full_theta, 'mean': full_theta}
    cat_theta_stats = _box_stats(cat_thetas) or {'min': 0.0, 'q1': 0.0, 'med': 0.0, 'q3': 0.0, 'max': 0.0, 'mean': 0.0}
    theta_diffs = [float(cat_theta) - full_theta for cat_theta in cat_thetas]
    return {
        'person_id': str(person_info.get('person_id', '')),
        'n_reps': n_reps,
        'full_length': full_length,
        'stop_se': float(stop_se),
        'full_summary': {'n': 1, 'mean_length': float(full_length), 'sd_length': 0.0},
        'full_theta_summary': summarize_mean_sd(full_theta_single),
        'cat_length_summary': summarize_mean_sd(cat_lengths),
        'cat_theta_summary': summarize_mean_sd(cat_thetas),
        'theta_diff_summary': summarize_mean_sd(theta_diffs),
        'full_length_stats': full_length_stats,
        'cat_length_stats': cat_length_stats,
        'full_theta_stats': full_theta_stats,
        'cat_theta_stats': cat_theta_stats,
        'length_ttest': onesample_ttest_against_constant(cat_lengths, full_length),
        'theta_diff_ttest': onesample_ttest_against_constant(cat_thetas, full_theta),
        'length_svg': make_length_efficiency_svg(full_length, cat_lengths),
        'theta_svg': make_theta_boxplot_svg(full_theta_single, cat_thetas),
        'actual_rows': actual_rows,
    }




def build_cat_noncat_n_comparison(state: dict) -> Optional[dict]:
    if str(state.get('mode', 'cat')) != 'compare_n':
        return None
    start_theta = float(state.get('start_theta', BANK.prior_mean))
    stop_se = float(state.get('stop_se', 0.32) or 0.32)
    person_infos = _person_infos_from_compare_n_state(state, random.Random(20260329))
    if not person_infos:
        return None
    full_lengths: List[float] = []
    full_thetas: List[float] = []
    cat_lengths: List[float] = []
    cat_thetas: List[float] = []
    selected_ids: List[str] = []
    for idx, person_info in enumerate(person_infos):
        score_map = dict(person_info.get('score_map', {}) or {})
        if not score_map:
            continue
        selected_ids.append(str(person_info.get('person_id', f'p{idx+1}')))
        full_run = _run_full_noncat_with_fixed_map(score_map, start_theta)
        full_lengths.append(float(full_run['length']))
        full_thetas.append(float(full_run['theta']))
        cap = max(1, int(full_run['length']))
        rng = random.Random(20260329 + int(person_info.get('row_index', idx)) + idx * 7919)
        cat_run = _run_cat_with_fixed_map(score_map, start_theta, stop_se, cap, rng, random_first=True)
        cat_lengths.append(float(cat_run['length']))
        cat_thetas.append(float(cat_run['theta']))
    if not cat_lengths or not full_lengths:
        return None
    full_len_summary = summarize_mean_sd(full_lengths)
    cat_len_summary = summarize_mean_sd(cat_lengths)
    full_theta_summary = summarize_mean_sd(full_thetas)
    cat_theta_summary = summarize_mean_sd(cat_thetas)
    length_paired = paired_ttest(cat_lengths, full_lengths)
    theta_paired = paired_ttest(cat_thetas, full_thetas)
    return {
        'n_persons': len(cat_lengths),
        'selected_person_ids': selected_ids,
        'selected_person_preview': ', '.join(selected_ids[:8]) + (' ...' if len(selected_ids) > 8 else ''),
        'stop_se': float(stop_se),
        'full_summary': {
            'n': int(len(full_lengths)),
            'mean_length': float(full_len_summary['mean']),
            'sd_length': float(full_len_summary['sd']),
        },
        'cat_length_summary': cat_len_summary,
        'full_length_stats': {
            'min': float(np.min(full_lengths)),
            'med': float(np.median(full_lengths)),
            'max': float(np.max(full_lengths)),
        },
        'cat_length_stats': {
            'min': float(np.min(cat_lengths)),
            'med': float(np.median(cat_lengths)),
            'max': float(np.max(cat_lengths)),
        },
        'length_ttest': length_paired,
        'full_theta_summary': full_theta_summary,
        'cat_theta_summary': cat_theta_summary,
        'full_theta_stats': {
            'min': float(np.min(full_thetas)),
            'med': float(np.median(full_thetas)),
            'max': float(np.max(full_thetas)),
        },
        'cat_theta_stats': {
            'min': float(np.min(cat_thetas)),
            'med': float(np.median(cat_thetas)),
            'max': float(np.max(cat_thetas)),
        },
        'theta_diff_ttest': theta_paired,
        'theta_diff_summary': {
            'mean': float(theta_paired.get('mean', 0.0)),
            'sd': float(theta_paired.get('sd', 0.0)),
        },
        'length_svg': make_length_boxplot_svg(full_lengths, cat_lengths),
        'theta_svg': make_theta_boxplot_svg(full_thetas, cat_thetas),
        'actual_rows': len(cat_lengths),
    }


def simulate_compare_n_session(language: str, start_theta: float, stop_se: float, compare_n_persons: int) -> dict:
    rng = random.Random()
    person_infos = _sample_n_original_person_infos(rng, compare_n_persons)
    if not person_infos:
        return simulate_compare_session(language=language, start_theta=start_theta, stop_se=stop_se, compare_n_reps=compare_n_persons)
    example_info = person_infos[0]
    score_map = dict(example_info.get('score_map', {}) or {})
    cap = len(score_map) if score_map else len(BANK.items)
    cat_run = _run_cat_with_fixed_map(score_map, start_theta, stop_se, cap, rng, random_first=True)
    return {
        'mode': 'compare_n',
        'max_items': cap,
        'requested_max_items': cap,
        'compare_n_persons': int(compare_n_persons),
        'compare_n_person_indices': [int(x.get('row_index', -1)) for x in person_infos],
        'compare_n_person_ids': [str(x.get('person_id', '')) for x in person_infos],
        'stop_se': float(stop_se),
        'start_theta': float(start_theta),
        'theta_range': 1.0,
        'start_item': 1,
        'language': language,
        'responses': list(cat_run['responses']),
        'history': list(cat_run['history']),
        'theta': float(cat_run['theta']),
        'se': float(cat_run['se']),
        'stop_reason': 'comparison_n_complete',
    }

def simulate_compare_session(language: str, start_theta: float, stop_se: float, compare_n_reps: int) -> dict:
    rng = random.Random()
    person_info = _sample_original_person_score_map(rng)
    score_map = dict(person_info.get('score_map', {}) or {})
    cap = len(score_map) if score_map else len(BANK.items)
    cat_run = _run_cat_with_fixed_map(score_map, start_theta, stop_se, cap, rng, random_first=True)
    return {
        'mode': 'compare',
        'max_items': cap,
        'requested_max_items': cap,
        'compare_n_reps': int(compare_n_reps),
        'compare_person_id': str(person_info.get('person_id', '')),
        'compare_person_index': int(person_info.get('row_index', -1)),
        'stop_se': float(stop_se),
        'start_theta': float(start_theta),
        'theta_range': 1.0,
        'start_item': 1,
        'language': language,
        'responses': [list(x) for x in cat_run['responses']],
        'history': list(cat_run['history']),
        'theta': float(cat_run['theta']),
        'se': float(cat_run['se']),
        'stop_reason': 'comparison_complete',
    }


app = Flask(__name__)
app.secret_key = SECRET_KEY
BANK = RaschCATBank(DEFAULT_BUNDLE)


def get_state() -> dict:
    return session.setdefault("cat_state", {})


def reset_state() -> None:
    session["cat_state"] = {}
    session.modified = True


def get_labels(language: str) -> Dict[str, str]:
    return LABELS[language if language in LABELS else "en"]


def resolve_link_href(raw_link: str) -> Tuple[str, bool]:
    raw = (raw_link or "").strip()
    if not raw:
        return "", False
    if re.match(r"^[a-z]+://", raw, re.I):
        return raw, raw.lower().endswith(IMG_EXTS)
    local_path = BANK.local_asset_path(raw)
    if not local_path:
        return "", False
    try:
        rel = local_path.relative_to(BANK.extract_dir)
    except ValueError:
        rel = local_path.relative_to(Path(__file__).parent)
    rel_path = str(rel).replace("\\", "/")
    return url_for("bundle_asset", asset_path=rel_path), rel_path.lower().endswith(IMG_EXTS)


def voice_part_texts(item: ItemRecord, language: str, item_number: int) -> Dict[str, str]:
    text_view = item.text_for(language)
    stem = str(text_view.get("stem") or text_view.get("full_text") or "")
    options = text_view.get("options") or {}
    if language == "zh":
        question_lead = f"第 {item_number} 題。"
        answer_prefix = "正確答案是"
        question_lines = [f"{question_lead} {stem}"]
    else:
        question_lead = f"Question {item_number}."
        answer_prefix = "The correct answer is"
        question_lines = [f"{question_lead} {stem}"]
    for lab, txt in options.items():
        question_lines.append(f"{lab}. {txt}")
    answer_text = str(options.get(item.key, "") or "")
    answer_line = f"{answer_prefix} {item.key}. {answer_text}" if answer_text else f"{answer_prefix} {item.key}."
    return {
        "question": " ".join(x.strip() for x in question_lines if x and x.strip()),
        "answer": answer_line.strip(),
    }


def synthesize_tts_path(text_value: str, language: str) -> Optional[Path]:
    payload = (text_value or "").strip()
    if not payload or gTTS is None:
        return None
    lang_code = "en" if language == "en" else "zh-TW"
    digest = hashlib.sha256((lang_code + "\n" + payload).encode("utf-8")).hexdigest()
    out_path = TTS_CACHE_DIR / f"{digest}.mp3"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    tmp_path = TTS_CACHE_DIR / f"{digest}.tmp.mp3"
    try:
        tts = gTTS(text=payload, lang=lang_code, slow=False)
        with open(tmp_path, "wb") as fh:
            tts.write_to_fp(fh)
        if tmp_path.exists() and tmp_path.stat().st_size > 0:
            tmp_path.replace(out_path)
            return out_path
    except Exception:
        pass
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
    return None


@app.get("/bundle_asset/<path:asset_path>")
def bundle_asset(asset_path: str):
    asset_path = asset_path.replace("\\", "/")
    for root in (BANK.extract_dir, Path(__file__).parent):
        fp = root / asset_path
        if fp.exists() and fp.is_file():
            return send_from_directory(str(root), asset_path)
    abort(404)


@app.get("/voice_tts")
def voice_tts():
    part = str(request.args.get("part", "question")).strip().lower()
    if part not in {"question", "answer"}:
        abort(404)
    item_id = str(request.args.get("item_id", "")).strip()
    if not item_id:
        abort(404)
    item = BANK.item_lookup.get(item_id)
    if item is None:
        abort(404)
    language = str(request.args.get("lang", "en")).strip().lower()
    if language not in {"zh", "en"}:
        language = "en"
    try:
        display_index = int(request.args.get("n", 1))
    except Exception:
        display_index = 1
    display_index = max(1, display_index)
    text_map = voice_part_texts(item, language, display_index)
    audio_path = synthesize_tts_path(text_map.get(part, ""), language)
    if not audio_path:
        return ("Server TTS unavailable", 503)
    resp = send_file(audio_path, mimetype="audio/mpeg", as_attachment=False, download_name=f"{part}.mp3")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["X-Voice-Item"] = item_id
    resp.headers["X-Voice-Part"] = part
    return resp


@app.get("/beep.mp3")
def beep_mp3():
    for candidate in ("pic/beep.mp3", "pic/beep.wav"):
        for root in (BANK.extract_dir, Path(__file__).parent):
            fp = root / candidate
            if fp.exists() and fp.is_file():
                mimetype = "audio/mpeg" if fp.suffix.lower() == ".mp3" else "audio/wav"
                return send_file(fp, mimetype=mimetype, as_attachment=False, download_name=fp.name)
    abort(404)




def mode_name_from_key(mode: str, labels: Dict[str, str]) -> str:
    if mode == "linear":
        return labels["mode_linear_short"]
    if mode == "voice":
        return labels["mode_voice_short"]
    if mode == "demo":
        return labels["mode_demo_short"]
    if mode == "compare":
        return labels["mode_compare_short"]
    if mode == "compare_n":
        return labels["mode_compare_n_short"]
    return labels["mode_cat_short"]


def stop_reason_text(reason: str, language: str) -> str:
    reason = str(reason or "finished")
    mapping = {
        "max_items": {"en": "maximum items reached", "zh": "已達最大題數"},
        "target_se": {"en": "target SE reached", "zh": "已達目標 SE"},
        "all_items": {"en": "all items completed", "zh": "所有題目已完成"},
        "demo_complete": {"en": "quick demo completed", "zh": "快速示範已完成"},
        "comparison_complete": {"en": "CAT vs non-CAT comparison completed", "zh": "CAT vs non-CAT 比較已完成"},
        "comparison_n_complete": {"en": "CAT vs non-CAT(n) comparison completed", "zh": "CAT vs non-CAT(n) 比較已完成"},
        "finished": {"en": "finished", "zh": "完成"},
    }
    return mapping.get(reason, {}).get(language, reason)


def simulate_demo_session(language: str, start_theta: float, requested_max_items: int) -> dict:
    rng = random.SystemRandom()
    max_items = max(1, min(20, len(BANK.items)))
    responses: List[Tuple[str, int]] = []
    history: List[Dict[str, object]] = []
    theta, se, _ = BANK.posterior([], start_theta=start_theta)
    while len(responses) < max_items:
        used_ids = [item_id for item_id, _ in responses]
        item = BANK.select_next_item(used_ids, theta)
        if item is None:
            break
        theta_before = float(theta)
        p = float(np.clip(BANK.probability(theta_before, item.delta), 1e-8, 1 - 1e-8))
        option_keys = list((item.options_zh or {}).keys()) or OPTION_LABELS[:max(2, len((item.options_zh or {})))]
        answer = str(rng.choice(option_keys)).upper()
        score = int(answer == item.key)
        responses.append((item.item_id, score))
        theta, se, _ = BANK.posterior(responses, start_theta=start_theta)
        zstd = (score - p) / math.sqrt(max(p * (1.0 - p), 1e-8))
        link_href, _ = resolve_link_href(item.link)
        history.append({
            "item_id": item.item_id,
            "no": item.no,
            "delta": float(item.delta),
            "answer": answer,
            "correct_tf": "TRUE" if score == 1 else "FALSE",
            "theta_before": theta_before,
            "theta": float(theta),
            "se": float(se),
            "zstd": float(zstd),
            "link_href": link_href,
        })
    return {
        "mode": "demo",
        "max_items": max_items,
        "requested_max_items": requested_max_items,
        "stop_se": 0.0,
        "start_theta": start_theta,
        "theta_range": 1.0,
        "start_item": 1,
        "language": language,
        "responses": [list(x) for x in responses],
        "history": history,
        "theta": float(theta),
        "se": float(se),
        "stop_reason": "demo_complete",
    }


def make_trend_svg(history: List[dict]) -> str:
    if not history:
        return _svg_wrap(900, 260, '<text x="20" y="40">No CAT trend data available.</text>')
    width, height = 920, 320
    left, right, top, bottom = 58, 20, 24, 42
    x0, y0 = left, height - bottom
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = list(range(1, len(history) + 1))
    theta_vals = [float(row.get('theta', 0.0)) for row in history]
    delta_vals = [float(row.get('delta', 0.0)) for row in history]
    zstd_vals = [float(row.get('zstd', 0.0)) for row in history]
    all_vals = theta_vals + delta_vals + zstd_vals + [-2.0, 2.0]
    ymin = min(all_vals)
    ymax = max(all_vals)
    if ymax <= ymin:
        ymax = ymin + 1.0
    pad = max(0.4, (ymax - ymin) * 0.08)
    ymin -= pad
    ymax += pad
    def x_map(v: float) -> float:
        if len(xs) == 1:
            return x0 + plot_w / 2
        return x0 + (v - 1) / max(1, len(xs) - 1) * plot_w
    def y_map(v: float) -> float:
        return y0 - (v - ymin) / (ymax - ymin) * plot_h
    def series_path(vals: List[float]) -> str:
        pts = [f"{x_map(i+1):.1f},{y_map(v):.1f}" for i, v in enumerate(vals)]
        return "M " + " L ".join(pts) if pts else ""
    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
             f'<line x1="{x0}" y1="{y0}" x2="{x0 + plot_w}" y2="{y0}" stroke="#444"/>',
             f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{y0}" stroke="#444"/>']
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        yv = ymin + (ymax - ymin) * frac
        yy = y_map(yv)
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + plot_w}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x0 - 8}" y="{yy + 4:.1f}" text-anchor="end" font-size="11">{yv:.2f}</text>')
    for idx in xs:
        xx = x_map(idx)
        parts.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0 + 5}" stroke="#444"/>')
        parts.append(f'<text x="{xx:.1f}" y="{y0 + 18}" text-anchor="middle" font-size="11">{idx}</text>')
    parts.append(f'<text x="{x0 + plot_w/2:.1f}" y="{height - 10}" text-anchor="middle" font-size="12">CAT item number</text>')
    parts.append(f'<text x="18" y="{top + plot_h/2:.1f}" transform="rotate(-90 18 {top + plot_h/2:.1f})" text-anchor="middle" font-size="12">Theta / Item difficulty / ZSTD</text>')
    colors = [(theta_vals, '#2563eb', 'Theta'), (delta_vals, '#dc2626', 'Item difficulty'), (zstd_vals, '#059669', 'ZSTD')]
    for vals, color, label in colors:
        path_d = series_path(vals)
        if path_d:
            parts.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for i, v in enumerate(vals, start=1):
            parts.append(f'<circle cx="{x_map(i):.1f}" cy="{y_map(v):.1f}" r="3.2" fill="{color}"/>')
    legend_x = x0 + 8
    legend_y = top + 8
    for j, (_, color, label) in enumerate(colors):
        ly = legend_y + j * 18
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 22}" y2="{ly}" stroke="{color}" stroke-width="2.4"/>')
        parts.append(f'<text x="{legend_x + 28}" y="{ly + 4}" font-size="11">{label}</text>')
    return _svg_wrap(width, height, ''.join(parts))


@app.get("/")
def index():
    summary = {
        "n_items": len(BANK.items),
        "prior_mean": BANK.prior_mean,
        "prior_sd": BANK.prior_sd,
        "n_persons": len(getattr(BANK, "original_response_df", pd.DataFrame())),
    }
    summary["n_sim_max"] = max(summary["n_items"], summary["n_persons"] or 0, 2)
    labels = get_labels("en")
    home_wright_svg = make_home_wrightmap_svg(BANK.person_distribution, BANK.item_fit_df)
    return render_template_string(
        HOME_TMPL,
        title=APP_TITLE,
        bundle_name=BANK.bundle_path.name,
        summary=summary,
        readme_text=BANK.readme_text,
        labels=labels,
        home_wright_svg=home_wright_svg,
    )


@app.post("/start")
def start_test():
    requested_max_items = int(request.form.get("max_items", 20))
    stop_se = float(request.form.get("stop_se", 0.32))
    start_theta = float(request.form.get("start_theta", BANK.prior_mean))
    theta_range_raw = str(request.form.get("theta_range", "1")).strip()
    try:
        theta_range = float(theta_range_raw)
    except Exception:
        theta_range = 1.0
    language = str(request.form.get("language", "zh")).strip().lower()
    start_item = int(request.form.get("start_item", 1))
    compare_n_reps = int(request.form.get("sim_cat_number", 20))
    mode = str(request.form.get("mode", "cat")).strip().lower()
    if language not in {"zh", "en"}:
        language = "zh"
    if mode not in {"cat", "linear", "voice", "demo", "compare", "compare_n"}:
        mode = "cat"

    start_item = max(1, min(start_item, len(BANK.items)))
    compare_n_reps = max(2, compare_n_reps)
    allowed_theta_ranges = [1.0, 1.5, 2.0, 2.5, 3.0]
    theta_range = min(allowed_theta_ranges, key=lambda x: abs(x - float(theta_range)))
    theta, se, _ = BANK.posterior([], start_theta=start_theta)
    if mode == "demo":
        session["cat_state"] = simulate_demo_session(language=language, start_theta=start_theta, requested_max_items=requested_max_items)
        session.modified = True
        return redirect(url_for("show_result"))
    if mode == "compare":
        session["cat_state"] = simulate_compare_session(language=language, start_theta=start_theta, stop_se=stop_se, compare_n_reps=compare_n_reps)
        session.modified = True
        return redirect(url_for("show_result"))
    if mode == "compare_n":
        session["cat_state"] = simulate_compare_n_session(language=language, start_theta=start_theta, stop_se=stop_se, compare_n_persons=compare_n_reps)
        session.modified = True
        return redirect(url_for("show_result"))
    if mode == "voice":
        voice_items = BANK.sample_voice_items(center_theta=start_theta, theta_range=theta_range, n_items=requested_max_items)
        if not voice_items:
            return redirect(url_for("index"))
        session["cat_state"] = {
            "mode": mode,
            "max_items": len(voice_items),
            "requested_max_items": requested_max_items,
            "stop_se": 0.0,
            "start_theta": start_theta,
            "theta_range": theta_range,
            "start_item": start_item,
            "language": language,
            "responses": [],
            "history": [],
            "voice_item_ids": [item.item_id for item in voice_items],
            "theta": start_theta,
            "se": se,
            "stop_reason": "",
        }
        session.modified = True
        return redirect(url_for("show_voice"))
    if mode == "linear":
        first_item = BANK.next_linear_item([], start_no=start_item)
        max_items = len(BANK.items)
        stop_se = 0.0
    else:
        first_item = BANK.select_next_item([], theta)
        max_items = requested_max_items
    if first_item is None:
        return redirect(url_for("index"))

    session["cat_state"] = {
        "mode": mode,
        "max_items": max_items,
        "requested_max_items": requested_max_items,
        "stop_se": stop_se,
        "start_theta": start_theta,
        "theta_range": theta_range,
        "start_item": start_item,
        "language": language,
        "responses": [],
        "history": [],
        "current_item": first_item.item_id,
        "theta": theta,
        "se": se,
        "stop_reason": "",
    }
    session.modified = True
    return redirect(url_for("show_item"))


@app.get("/item")
def show_item():
    state = get_state()
    if not state or "current_item" not in state:
        return redirect(url_for("index"))
    language = state.get("language", "zh")
    labels = get_labels(language)
    item = BANK.item_lookup[state["current_item"]]
    text_view = item.text_for(language)
    link_href, is_image = resolve_link_href(item.link)
    mode = state.get("mode", "cat")
    progress = {
        "answered": len(state.get("responses", [])),
        "max_items": state["max_items"],
        "theta": state["theta"],
        "se": state["se"],
        "info_line": labels["sel_info_cat"] if mode == "cat" else labels["sel_info_linear"],
    }
    item_view = {
        "item_id": item.item_id,
        "no": item.no,
        "full_text": text_view["full_text"],
        "stem": text_view["stem"],
        "options": text_view["options"],
        "link_href": link_href,
        "is_image_link": is_image,
    }
    language_name = labels["lang_zh"] if language == "zh" else labels["lang_en"]
    mode_name = mode_name_from_key(mode, labels)
    return render_template_string(ITEM_TMPL, title=APP_TITLE, item=item_view, progress=progress, labels=labels, language_name=language_name, mode_name=mode_name)


@app.get("/voice")
def show_voice():
    state = get_state()
    if not state or state.get("mode") != "voice":
        return redirect(url_for("index"))
    voice_ids = [str(x) for x in state.get("voice_item_ids", [])]
    if not voice_ids:
        return redirect(url_for("index"))
    language = state.get("language", "en")
    labels = get_labels(language)
    payload = []
    for item_id in voice_ids:
        item = BANK.item_lookup.get(item_id)
        if not item:
            continue
        text_view = item.text_for(language)
        options_dict = text_view.get("options") or {}
        options = [{"label": str(k), "text": str(v)} for k, v in options_dict.items()]
        answer_text = str(options_dict.get(item.key, "") or "")
        link_href, is_image = resolve_link_href(item.link)
        payload.append({
            "item_id": item.item_id,
            "no": item.no,
            "delta": float(item.delta),
            "full_text": str(text_view.get("full_text") or ""),
            "stem": str(text_view.get("stem") or text_view.get("full_text") or ""),
            "options": options,
            "key": item.key,
            "answer_text": answer_text,
            "link_href": link_href,
            "is_image_link": is_image,
        })
    speech_config = {
        "lang": "en-US" if language == "en" else "zh-TW",
        "language_code": language,
        "question_prefix": "Question" if language == "en" else "第",
        "answer_prefix": "The correct answer is" if language == "en" else "正確答案是",
        "item_label": labels["item"],
        "ready_status": labels["voice_status_ready"],
        "reading_status": "Reading item" if language == "en" else "正在朗讀第",
        "pause_status": "Pause before answer..." if language == "en" else "答案前停頓中…",
        "answer_status": "Reading correct answer..." if language == "en" else "正在朗讀正確答案…",
        "finished_status": "Finished item" if language == "en" else "已完成第",
        "stopped_status": "Audio stopped." if language == "en" else "語音已停止。",
        "paused_status": "Audio paused." if language == "en" else "語音已暫停。",
        "resumed_status": "Audio resumed." if language == "en" else "語音已繼續。",
        "done_title": labels["voice_done"],
        "done_body": labels["voice_done"],
        "tts_base_url": url_for("voice_tts"),
        "server_tts_enabled": bool(gTTS is not None),
        "server_engine_label": "Direct MP3 audio URL" if language == "en" else "直接 MP3 音檔網址",
        "browser_unavailable_status": "Browser TTS is unavailable on this device." if language == "en" else "此裝置未提供瀏覽器 TTS。",
        "touch_failed_status": "Touch action failed." if language == "en" else "手機觸控動作失敗。",
        "browser_engine_label": "Browser TTS fallback" if language == "en" else "瀏覽器 TTS 備援",
        "server_failed_status": "Server audio failed; switching to browser TTS." if language == "en" else "伺服器音檔失敗，改用瀏覽器 TTS。",
        "mobile_tap_status": "Tap Start audio cycle on mobile to play the direct MP3 URL." if language == "en" else "手機請按「開始語音循環」以播放直接 MP3 網址。",
    }
    language_name = labels["lang_zh"] if language == "zh" else labels["lang_en"]
    return render_template_string(
        VOICE_TMPL,
        title=APP_TITLE,
        labels=labels,
        items=payload,
        item_count=len(payload),
        requested_max_items=int(state.get("requested_max_items", len(payload))),
        start_theta=float(state.get("start_theta", BANK.prior_mean)),
        theta_range=float(state.get("theta_range", 1.0)),
        language_name=language_name,
        speech_config=speech_config,
    )


@app.get("/to_home")
def to_home():
    reset_state()
    return redirect(url_for("index"))


@app.post("/answer")
def submit_answer():
    state = get_state()
    if not state or "current_item" not in state:
        return redirect(url_for("index"))
    answer = str(request.form.get("answer", "")).strip().upper()
    item = BANK.item_lookup[state["current_item"]]
    score = int(answer == item.key)
    mode = state.get("mode", "cat")
    theta_before = float(state.get("theta", state.get("start_theta", BANK.prior_mean)))
    p_before = float(np.clip(BANK.probability(theta_before, item.delta), 1e-8, 1 - 1e-8))
    zstd = (score - p_before) / math.sqrt(max(p_before * (1.0 - p_before), 1e-8))

    responses = [tuple(x) for x in state.get("responses", [])]
    responses.append((item.item_id, score))
    theta, se, _ = BANK.posterior(responses, start_theta=state.get("start_theta", BANK.prior_mean))
    history = list(state.get("history", []))
    link_href, _ = resolve_link_href(item.link)
    history.append({
        "item_id": item.item_id,
        "no": item.no,
        "delta": item.delta,
        "answer": answer,
        "correct_tf": "TRUE" if score == 1 else "FALSE",
        "theta_before": theta_before,
        "theta": theta,
        "se": se,
        "zstd": zstd,
        "link_href": link_href,
    })

    stop_reason = ""
    if mode == "linear":
        if len(responses) >= len(BANK.items):
            stop_reason = "all_items"
    else:
        if len(responses) >= int(state["max_items"]):
            stop_reason = "max_items"
        elif se <= float(state["stop_se"]):
            stop_reason = "target_se"

    state["responses"] = [list(x) for x in responses]
    state["history"] = history
    state["theta"] = theta
    state["se"] = se
    state["stop_reason"] = stop_reason

    if stop_reason:
        session["cat_state"] = state
        session.modified = True
        return redirect(url_for("show_result"))

    used_ids = [i for i, _ in responses]
    if mode == "linear":
        next_item = BANK.next_linear_item(used_ids, start_no=state.get("start_item", 1))
    else:
        next_item = BANK.select_next_item(used_ids, theta)
    if next_item is None:
        state["stop_reason"] = "all_items"
        session["cat_state"] = state
        session.modified = True
        return redirect(url_for("show_result"))
    state["current_item"] = next_item.item_id
    session["cat_state"] = state
    session.modified = True
    return redirect(url_for("show_item"))


@app.get("/result")
def show_result():
    state = get_state()
    if not state or not state.get("history"):
        return redirect(url_for("index"))
    language = state.get("language", "zh")
    labels = get_labels(language)
    dashboard = build_dashboard_data(state)
    mode = state.get("mode", "cat")
    history_rows = []
    for row in state["history"]:
        row2 = dict(row)
        if "correct_tf" not in row2:
            item_obj = BANK.item_lookup.get(str(row2.get("item_id", "")))
            ans = str(row2.get("answer", "")).strip().upper()
            row2["correct_tf"] = "TRUE" if item_obj and ans == item_obj.key else "FALSE"
        history_rows.append(row2)
    result = {
        "theta": state["theta"],
        "se": state["se"],
        "percentile": BANK.percentile(state["theta"]),
        "n_answered": len(state["responses"]),
        "stop_reason": stop_reason_text(state.get("stop_reason", "finished"), language),
        "history": history_rows,
        "has_links": any(bool((row.get("link_href") or "").strip()) for row in history_rows),
        "mode_name": mode_name_from_key(mode, labels),
        **dashboard,
    }
    return render_template_string(RESULT_TMPL, title=APP_TITLE, result=result, labels=labels)


@app.get("/reset")
def reset():
    reset_state()
    return redirect(url_for("index"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
