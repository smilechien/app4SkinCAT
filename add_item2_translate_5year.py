#!/usr/bin/env python3
"""Add or fill an `item2` English translation column for the 5-year exam CSV.

Usage:
  python add_item2_translate_5year.py \
    --input 110_114_5year_item_key_with_symbols.csv \
    --output 110_114_5year_item_key_with_symbols_en.csv

Requirements:
  pip install pandas deep-translator

What this version changes:
  - Works with the 5-year file structure (e.g. year/no/key/link/item/item2).
  - Translates each row into `item2` while preserving the original `item` column.
  - Saves after every row so interrupted runs can be resumed.
  - Reads CSVs with common encodings used in Taiwan (utf-8-sig / utf-8 / cp950 / big5).
  - Translates line-by-line to better preserve numbered statements and answer-option lines.
  - Skips rows that already have `item2` content.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
from deep_translator import GoogleTranslator


ENCODINGS_TO_TRY = ["utf-8-sig", "utf-8", "cp950", "big5", "latin1"]

# Keep common circled numerals intact and normalize option markers.
CHOICE_MAP = {
    "（A）": "(A)", "（B）": "(B)", "（C）": "(C)", "（D）": "(D)", "（E）": "(E)",
    "（F）": "(F)", "（G）": "(G)", "（H）": "(H)", "（I）": "(I)", "（J）": "(J)",
    "Ａ.": "A.", "Ｂ.": "B.", "Ｃ.": "C.", "Ｄ.": "D.", "Ｅ.": "E.",
}

CIRCLED_NUMS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ENCODINGS_TO_TRY:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"Unable to read CSV: {path}\nLast error: {last_err}")



def normalize_zh_text(text: str) -> str:
    """Normalize punctuation/spacing while preserving line breaks."""
    text = "" if pd.isna(text) else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u3000", " ")

    for zh, en in CHOICE_MAP.items():
        text = text.replace(zh, en)

    # Normalize punctuation but keep newlines.
    text = text.replace("？", "?")
    text = text.replace("，", ", ")
    text = text.replace("；", "; ")
    text = text.replace("：", ": ")
    text = text.replace("。", ". ")

    lines = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"\s+", " ", line)
        # Ensure answer-option line spacing is consistent.
        line = re.sub(r"\b([A-J])\.\s*", r"\1. ", line)
        lines.append(line)
    return "\n".join(lines).strip()



def postprocess_en(text: str) -> str:
    """Clean up translated English while preserving line structure."""
    text = "" if text is None else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    out_lines = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"\s+", " ", line)
        line = re.sub(r"\(\s*([A-Z])\s*\)", r"(\1)", line)
        line = line.replace(" .", ".").replace(" ,", ",").replace(" ;", ";").replace(" :", ":")
        line = re.sub(r"\b([A-J])\.\s*", r"\1. ", line)
        out_lines.append(line)
    return "\n".join(out_lines).strip()



def split_long_text(text: str, max_chars: int = 4000) -> list[str]:
    """Split long passages into smaller chunks for web translators."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    # Prefer splitting by line, then by sentence punctuation.
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        candidate = f"{current}\n{line}".strip() if current else line
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            flush()

        if len(line) <= max_chars:
            current = line
            continue

        parts = re.split(r"(?<=[.!?;:])\s+", line)
        sub = ""
        for part in parts:
            candidate2 = f"{sub} {part}".strip() if sub else part
            if len(candidate2) <= max_chars:
                sub = candidate2
            else:
                if sub:
                    chunks.append(sub.strip())
                sub = part
        if sub.strip():
            current = sub.strip()

    flush()
    return chunks



def translate_block(translator: GoogleTranslator, text: str, retries: int = 4, sleep_sec: float = 2.0) -> str:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            out = translator.translate(text)
            if not out or not str(out).strip():
                raise RuntimeError("empty translation returned")
            return str(out)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(sleep_sec * attempt)
    raise RuntimeError(f"translation failed after {retries} retries: {last_err}")



def translate_text(translator: GoogleTranslator, text: str, retries: int = 4, sleep_sec: float = 2.0) -> str:
    cleaned = normalize_zh_text(text)
    chunks = split_long_text(cleaned, max_chars=4000)
    translated_chunks = []
    for chunk in chunks:
        translated_chunks.append(translate_block(translator, chunk, retries=retries, sleep_sec=sleep_sec))
        time.sleep(0.5)
    merged = "\n".join(translated_chunks)
    return postprocess_en(merged)



def load_or_init_output(input_path: Path, output_path: Path) -> pd.DataFrame:
    if output_path.exists():
        df_out = read_csv_flexible(output_path)
        if "item2" not in df_out.columns:
            df_out["item2"] = ""
        return df_out

    df = read_csv_flexible(input_path)
    required = {"no", "key", "item"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if "item2" not in df.columns:
        df["item2"] = ""
    return df



def row_label(df: pd.DataFrame, idx: int) -> str:
    parts = []
    for col in ("year", "no", "key"):
        if col in df.columns:
            parts.append(f"{col}={df.at[idx, col]}")
    return ", ".join(parts) if parts else f"row={idx + 1}"



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output CSV with item2 column")
    parser.add_argument("--start-row", type=int, default=1, help="1-based row number to start/resume from")
    parser.add_argument("--source-lang", default="zh-TW", help="Translator source language")
    parser.add_argument("--target-lang", default="en", help="Translator target language")
    parser.add_argument("--sleep", type=float, default=0.8, help="Sleep seconds after each completed row")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = load_or_init_output(input_path, output_path)
    translator = GoogleTranslator(source=args.source_lang, target=args.target_lang)

    start_idx = max(0, args.start_row - 1)
    for idx in range(start_idx, len(df)):
        current = "" if pd.isna(df.at[idx, "item2"]) else str(df.at[idx, "item2"]).strip()
        if current:
            continue

        zh = df.at[idx, "item"]
        print(f"Translating row {idx + 1}/{len(df)} ({row_label(df, idx)})...", flush=True)
        try:
            df.at[idx, "item2"] = translate_text(translator, zh)
        except Exception as e:  # noqa: BLE001
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"\nStopped at row {idx + 1}: {e}", file=sys.stderr)
            print("Progress was saved. Re-run with --start-row to resume.", file=sys.stderr)
            return 1

        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        time.sleep(args.sleep)

    print(f"Done. Saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
