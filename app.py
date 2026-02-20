"""
app.py  –  Whisper Recorder  (Eel Desktop App)
======================================================
依存関係:
    pip install -r requirements.txt

起動:
    python app.py

動作:
    - Alt 系ホットキー長押し → 録音開始
    - Alt 系ホットキー離す   → Whisper で文字起こし → GPT-4o-mini で整形 → アクティブウィンドウへ自動ペースト
    - タスクトレイ常駐で動作継続
"""

from __future__ import annotations

import ctypes
import errno
import io
import json
import os
import re
import sys
import tempfile
import threading
import time
import wave

import eel
import keyboard
import numpy as np
import pyautogui
import pyperclip
from dotenv import load_dotenv
from PIL import Image, ImageDraw
import pystray

# ------------------------------------------------------------------
# 設定
# ------------------------------------------------------------------
load_dotenv()

SAMPLE_RATE      = 16_000   # Whisper 推奨
HOTKEY           = "right alt"
WINDOW_SIZE      = (1024, 680)
DICTIONARY_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dictionary.json")
SNIPPETS_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snippets.json")
HISTORY_FILE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.json")
SETTINGS_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
_CHROME_DATA_DIR = os.path.join(tempfile.gettempdir(), "whisper_recorder_chrome")
_MUTEX_NAME      = "Global\\WhisperRecorderSingleInstance"

_DEFAULT_SETTINGS = {
    "language": "ja",
    "history_retention_days": 30,
    "hotkey": "right alt",
}

# ------------------------------------------------------------------
# 二重起動防止 (Windows Named Mutex)
# ------------------------------------------------------------------
_instance_mutex = None
_ERROR_ALREADY_EXISTS = 183


def _acquire_instance_lock() -> bool:
    """Windows Named Mutex でプロセス単位の排他を実現する。
    既に別プロセスが Mutex を保持していれば False を返す。"""
    global _instance_mutex
    if sys.platform != "win32":
        return True
    kernel32 = ctypes.windll.kernel32
    _instance_mutex = kernel32.CreateMutexW(None, True, _MUTEX_NAME)
    if kernel32.GetLastError() == _ERROR_ALREADY_EXISTS:
        kernel32.CloseHandle(_instance_mutex)
        _instance_mutex = None
        return False
    return True


# ------------------------------------------------------------------
# Win32 キー状態ポーリング (AltGr 問題の回避策)
# ------------------------------------------------------------------
_VK_MENU     = 0x12   # Alt (generic)
_VK_LMENU    = 0xA4   # Left Alt
_VK_RMENU    = 0xA5   # Right Alt
_VK_RCONTROL = 0xA3   # Right Ctrl
_ALT_VKS     = (_VK_RMENU, _VK_MENU, _VK_LMENU)


def _is_vk_pressed(vk: int) -> bool:
    """Win32 GetAsyncKeyState で物理キーの押下状態を取得"""
    if sys.platform != "win32":
        return False
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def _get_hotkey_vks() -> tuple[int, ...]:
    hotkey = _normalize_hotkey(HOTKEY)
    if hotkey == "left alt":
        return (_VK_LMENU,)
    if hotkey == "either alt":
        return _ALT_VKS
    return (_VK_RMENU,)


def _is_hotkey_pressed() -> bool:
    """設定されたホットキーの押下状態を返す"""
    return any(_is_vk_pressed(vk) for vk in _get_hotkey_vks())

# ------------------------------------------------------------------
# カスタム辞書
# ------------------------------------------------------------------

def _load_dictionary() -> list[dict]:
    if not os.path.exists(DICTIONARY_FILE):
        return []
    try:
        with open(DICTIONARY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_dictionary(entries: list[dict]) -> None:
    with open(DICTIONARY_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def _get_dictionary_words() -> list[str]:
    """辞書に登録された word のリストを返す"""
    return [e["word"] for e in _load_dictionary() if e.get("word")]


# ------------------------------------------------------------------
# スニペット（定型文）
# ------------------------------------------------------------------

def _load_snippets() -> list[dict]:
    if not os.path.exists(SNIPPETS_FILE):
        return []
    try:
        with open(SNIPPETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_snippets(entries: list[dict]) -> None:
    with open(SNIPPETS_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


_NORMALIZE_RE = re.compile(r"[\s\u3000、。，．,.!！?？・：:；;…\-\u2010-\u2015\u2212\uff0d\"'()（）「」『』【】\[\]{}]")


def _normalize_for_snippet(text: str) -> str:
    """句読点・空白・記号を除去して小文字化した比較用文字列を返す"""
    return _NORMALIZE_RE.sub("", text).lower()


def _match_snippet(transcript: str) -> str | None:
    """transcript がスニペットのキーワードと一致すればその定型文を返す"""
    normalized = _normalize_for_snippet(transcript)
    if not normalized:
        return None
    for entry in _load_snippets():
        kw = entry.get("keyword", "")
        if _normalize_for_snippet(kw) == normalized:
            return entry.get("text", "")
    return None


# ------------------------------------------------------------------
# 設定
# ------------------------------------------------------------------

_SUPPORTED_LANGUAGES = {"ja", "en", "zh"}
_SUPPORTED_HOTKEYS = {"right alt", "left alt", "either alt"}


def _normalize_language(value: str | None) -> str:
    v = (value or "").strip().lower()
    if v in _SUPPORTED_LANGUAGES:
        return v
    return _DEFAULT_SETTINGS["language"]


def _normalize_history_retention_days(value) -> int | None:
    if value in (None, "", "forever"):
        return None
    try:
        days = int(value)
    except (TypeError, ValueError):
        return _DEFAULT_SETTINGS["history_retention_days"]
    if days <= 0:
        return None
    if days in (7, 30):
        return days
    return _DEFAULT_SETTINGS["history_retention_days"]


def _normalize_hotkey(value: str | None) -> str:
    v = (value or "").strip().lower()
    if v in _SUPPORTED_HOTKEYS:
        return v
    return _DEFAULT_SETTINGS["hotkey"]


def _normalize_settings(raw: dict | None) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    return {
        "language": _normalize_language(raw.get("language")),
        "history_retention_days": _normalize_history_retention_days(raw.get("history_retention_days")),
        "hotkey": _normalize_hotkey(raw.get("hotkey")),
    }


def _load_settings() -> dict:
    if not os.path.exists(SETTINGS_FILE):
        return dict(_DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return dict(_DEFAULT_SETTINGS)
    return _normalize_settings(raw)


def _save_settings(settings: dict) -> None:
    normalized = _normalize_settings(settings)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------
# 履歴
# ------------------------------------------------------------------

def _load_history() -> list[dict]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(raw, list):
        return []

    entries: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        ts = item.get("timestamp")
        try:
            ts_value = float(ts)
        except (TypeError, ValueError):
            ts_value = time.time()
        entries.append({"text": text, "timestamp": ts_value})
    return entries


def _save_history(entries: list[dict]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def _apply_history_retention(entries: list[dict]) -> list[dict]:
    settings = _load_settings()
    retention_days = settings.get("history_retention_days")
    if retention_days is None:
        filtered = entries
    else:
        cutoff = time.time() - (int(retention_days) * 24 * 60 * 60)
        filtered = [e for e in entries if float(e.get("timestamp", 0)) >= cutoff]
    return sorted(filtered, key=lambda e: float(e.get("timestamp", 0)), reverse=True)


def _compact_history() -> list[dict]:
    compacted = _apply_history_retention(_load_history())
    _save_history(compacted)
    return compacted


def _append_history(text: str) -> list[dict]:
    content = text.strip()
    if not content:
        return _compact_history()
    entries = _load_history()
    entries.append({"text": content, "timestamp": time.time()})
    compacted = _apply_history_retention(entries)
    _save_history(compacted)
    return compacted


# ------------------------------------------------------------------
# 指示プレフィックス検出
# ------------------------------------------------------------------
_PREFIX_INSTRUCTIONS: list[tuple[str, str]] = [
    ("英語にして",       "以下のテキストを自然な英語に翻訳してください。整形した英語のみ出力してください。"),
    ("英語で",           "以下のテキストを自然な英語に翻訳してください。整形した英語のみ出力してください。"),
    ("英訳して",         "以下のテキストを自然な英語に翻訳してください。整形した英語のみ出力してください。"),
    ("箇条書きで",       "以下のテキストを箇条書き形式に変換してください。箇条書きのみ出力してください。"),
    ("丁寧なメール文にして", "以下のテキストを丁寧なビジネスメール文に変換してください。メール文のみ出力してください。"),
    ("メール文にして",   "以下のテキストを丁寧なビジネスメール文に変換してください。メール文のみ出力してください。"),
    ("要約して",         "以下のテキストを簡潔に要約してください。要約のみ出力してください。"),
    ("敬語にして",       "以下のテキストを丁寧な敬語に変換してください。変換後のテキストのみ出力してください。"),
    ("カジュアルにして", "以下のテキストをカジュアルな口語体に変換してください。変換後のテキストのみ出力してください。"),
]
_PREFIX_SEPARATORS = ("：", ":", "、", "。", " ", "　", ",", ".", "")


def _detect_prefix(text: str) -> tuple[str | None, str]:
    """テキスト冒頭から指示プレフィックスを検出し (instruction, remaining_text) を返す。
    見つからなければ (None, 元テキスト)。"""
    stripped = text.strip()
    for prefix, instruction in _PREFIX_INSTRUCTIONS:
        for sep in _PREFIX_SEPARATORS:
            candidate = prefix + sep
            if stripped.startswith(candidate):
                remaining = stripped[len(candidate):].strip()
                if remaining:
                    return instruction, remaining
    return None, text


# ------------------------------------------------------------------
# アプリ状態
# ------------------------------------------------------------------
_recording     = False
_rewrite_mode  = False
_selected_text = ""
_audio_frames: list[np.ndarray] = []
_tray_icon: pystray.Icon | None = None
_eel_window_open = False

# ------------------------------------------------------------------
# eel 初期化 (main() 内で実行。モジュールレベルだと Windows の再インポートで二重起動する)
# ------------------------------------------------------------------


# ==================================================================
# タスクトレイアイコン
# ==================================================================

def _make_tray_image(recording: bool = False) -> Image.Image:
    """シンプルなマイクアイコンを PIL で生成"""
    img  = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    color = (239, 68, 68) if recording else (59, 130, 246)   # 録音中は赤
    # マイク本体
    draw.rounded_rectangle([20, 4, 44, 36], radius=12, fill=color)
    # スタンド
    draw.rectangle([29, 36, 35, 50], fill=color)
    # ベース
    draw.arc([16, 44, 48, 58], start=0, end=180, fill=color, width=4)
    return img


def _tray_show_window(icon: pystray.Icon, item: pystray.MenuItem) -> None:
    """トレイ → ウィンドウを前面に"""
    try:
        eel.js_show_notification("ウィンドウを表示します")()
    except Exception:
        pass


def _tray_quit(icon: pystray.Icon, item: pystray.MenuItem) -> None:
    icon.stop()
    sys.exit(0)


def _start_tray() -> None:
    global _tray_icon
    menu = pystray.Menu(
        pystray.MenuItem("ウィンドウを表示", _tray_show_window),
        pystray.MenuItem("終了", _tray_quit),
    )
    _tray_icon = pystray.Icon(
        "whisper_recorder",
        _make_tray_image(),
        "音声入力ツール（Right Alt で録音）",
        menu,
    )
    _tray_icon.run()          # ブロッキング → デーモンスレッドで呼ぶ


def _update_tray_icon(recording: bool) -> None:
    """録音状態によってトレイアイコンの色を変更"""
    if _tray_icon:
        try:
            _tray_icon.icon = _make_tray_image(recording)
        except Exception:
            pass


# ==================================================================
# 録音
# ==================================================================

def _recording_thread() -> None:
    import sounddevice as sd
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
            while _recording:
                chunk, _ = stream.read(1024)
                _audio_frames.append(chunk.copy())
    except Exception as e:
        print(f"[App] マイク録音エラー: {e}")
        _safe_eel_call("js_show_notification", f"❌ マイクエラー: {e}")


def _do_start_recording() -> None:
    global _recording, _audio_frames
    if _recording:
        return
    _recording    = True
    _audio_frames = []
    print("[App] 録音開始...")
    _update_tray_icon(True)
    _safe_eel_call("js_set_recording_state", True)
    _safe_eel_call("js_show_notification", "🎙️ 録音中...")
    threading.Thread(target=_recording_thread, daemon=True).start()


def _do_stop_recording() -> None:
    global _recording
    if not _recording:
        return
    _recording = False
    time.sleep(0.15)
    print("[App] 録音停止")
    _update_tray_icon(False)
    _safe_eel_call("js_set_recording_state", False)
    _safe_eel_call("js_show_notification", "⏳ 文字起こし中...")
    threading.Thread(target=_process_audio, daemon=True).start()


# ==================================================================
# リライト録音 (right ctrl)
# ==================================================================

def _do_start_rewrite_recording() -> None:
    global _recording, _rewrite_mode, _selected_text, _audio_frames
    if _recording:
        return

    pyautogui.hotkey("ctrl", "c")
    time.sleep(0.1)
    _selected_text = pyperclip.paste()
    if not _selected_text.strip():
        print("[App] 選択テキストが空のためリライトを中止")
        _safe_eel_call("js_show_notification", "⚠️ テキストが選択されていません")
        return

    _rewrite_mode = True
    _recording    = True
    _audio_frames = []
    print(f"[App] リライト録音開始 (選択テキスト: {_selected_text[:40]}...)")
    _update_tray_icon(True)
    _safe_eel_call("js_set_recording_state", True)
    _safe_eel_call("js_show_notification", "🎙️ 音声指示を録音中...")
    threading.Thread(target=_recording_thread, daemon=True).start()


def _do_stop_rewrite_recording() -> None:
    global _recording, _rewrite_mode
    if not _recording:
        return
    _recording = False
    time.sleep(0.15)
    print("[App] リライト録音停止")
    _update_tray_icon(False)
    _safe_eel_call("js_set_recording_state", False)
    _safe_eel_call("js_show_notification", "⏳ リライト処理中...")
    threading.Thread(target=_process_rewrite_audio, daemon=True).start()


def _process_rewrite_audio() -> None:
    global _rewrite_mode
    try:
        if not _audio_frames:
            _safe_eel_call("js_show_notification", "❌ 音声が検出されませんでした")
            return

        instruction = _transcribe()
        if not instruction:
            _safe_eel_call("js_show_notification", "❌ 音声指示の文字起こしに失敗しました")
            return
        print(f"[Whisper] リライト指示: {instruction}")

        rewritten = _rewrite_with_gpt(_selected_text, instruction)
        print(f"[GPT]     リライト結果: {rewritten[:80]}")

        pyperclip.copy(rewritten)
        time.sleep(0.05)
        pyautogui.hotkey("ctrl", "v")

        _append_history(rewritten)
        preview = rewritten[:50] + ("..." if len(rewritten) > 50 else "")
        _safe_eel_call("js_show_notification", f"✅ リライト完了: {preview}")
        _safe_eel_call("js_add_history", rewritten)
    finally:
        _rewrite_mode = False


def _rewrite_with_gpt(original: str, instruction: str) -> str:
    """GPT-4o-mini で選択テキストを音声指示に従ってリライト"""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return original

    system_prompt = (
        "あなたは優秀なテキスト編集アシスタントです。"
        "ユーザーから『元のテキスト』と『編集指示』が与えられます。"
        "指示に従って元のテキストを書き換えてください。"
        "出力は書き換えられたテキストのみを出力し、説明やコメントは一切含めないでください。"
    )
    dict_words = _get_dictionary_words()
    if dict_words:
        system_prompt += (
            "\n以下の専門用語・固有名詞を優先して使用してください: "
            + ", ".join(dict_words)
        )

    user_message = (
        f"【元のテキスト】\n{original}\n\n"
        f"【編集指示】\n{instruction}"
    )

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=2000,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[App] GPT リライトエラー: {e}")
        return original


# ==================================================================
# 音声処理パイプライン: Whisper → GPT-4o-mini → ペースト
# ==================================================================

def _process_audio() -> None:
    if not _audio_frames:
        _safe_eel_call("js_show_notification", "❌ 音声が検出されませんでした")
        return

    # ① Whisper で文字起こし
    transcript = _transcribe()
    if not transcript:
        _safe_eel_call("js_show_notification", "❌ 文字起こしに失敗しました")
        return
    print(f"[Whisper] {transcript}")

    # ②-a スニペット判定（完全一致 → GPT スキップ）
    snippet_text = _match_snippet(transcript)
    if snippet_text is not None:
        print(f"[Snippet] キーワード一致 → 定型文を出力")
        pyperclip.copy(snippet_text)
        time.sleep(0.05)
        pyautogui.hotkey("ctrl", "v")
        _append_history(snippet_text)
        preview = snippet_text[:50] + ("..." if len(snippet_text) > 50 else "")
        _safe_eel_call("js_show_notification", f"📋 スニペット: {preview}")
        _safe_eel_call("js_add_history", snippet_text)
        return

    # ②-b プレフィックス検出 → GPT-4o-mini でテキスト整形 or 変換
    prefix_instruction, body = _detect_prefix(transcript)
    if prefix_instruction:
        print(f"[Prefix]  検出: {prefix_instruction[:30]}... body={body[:40]}")
    formatted = _format_with_gpt(body, prefix_instruction)
    print(f"[GPT]     {formatted}")

    # ③ クリップボードにコピー → アクティブウィンドウへペースト
    pyperclip.copy(formatted)
    time.sleep(0.05)
    pyautogui.hotkey("ctrl", "v")

    # ④ UI に通知・履歴追加
    _append_history(formatted)
    preview = formatted[:50] + ("..." if len(formatted) > 50 else "")
    _safe_eel_call("js_show_notification", f"✅ {preview}")
    _safe_eel_call("js_add_history", formatted)


def _transcribe() -> str | None:
    """録音 PCM を WAV に変換し Whisper API へ送信"""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[App] OPENAI_API_KEY が未設定です")
        return None

    audio = np.concatenate(_audio_frames, axis=0).flatten()
    int16 = (audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(int16.tobytes())
    buf.seek(0)
    buf.name = "audio.wav"

    try:
        client = OpenAI(api_key=api_key)
        language = _load_settings().get("language", _DEFAULT_SETTINGS["language"])
        whisper_kwargs: dict = dict(model="whisper-1", file=buf, language=language)
        dict_words = _get_dictionary_words()
        if dict_words:
            whisper_kwargs["prompt"] = ", ".join(dict_words)
        resp = client.audio.transcriptions.create(**whisper_kwargs)
        return resp.text.strip()
    except Exception as e:
        print(f"[App] Whisper エラー: {e}")
        return None


def _format_with_gpt(text: str, prefix_instruction: str | None = None) -> str:
    """GPT-4o-mini で音声認識テキストを整形 or 変換する。
    prefix_instruction が指定されている場合はそれを最優先で適用する。"""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return text

    if prefix_instruction:
        system_prompt = (
            "あなたは優秀なテキスト変換AIです。\n"
            "ユーザーから与えられたテキストに対して、以下の指示を適用してください:\n"
            f"【指示】{prefix_instruction}\n"
            "出力は変換後のテキストのみを返してください。説明・コメントは一切不要です。"
        )
    else:
        system_prompt = (
            "あなたは音声認識テキストの整形AIです。\n"
            "入力テキストを以下のルールで整形してください:\n"
            "1. 不自然な繰り返しや言い間違いを修正\n"
            "2. 句読点を適切に追加\n"
            "3. 自然な日本語または英語の文章にする\n"
            "4. 内容や意味は変えない\n"
            "5. 整形した文章だけを返す（説明・コメント不要）"
        )

    dict_words = _get_dictionary_words()
    if dict_words:
        system_prompt += (
            "\n以下の専門用語・固有名詞を優先して使用してください: "
            + ", ".join(dict_words)
        )

    try:
        client = OpenAI(api_key=api_key)
        resp   = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": text},
            ],
            max_tokens=2000,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[App] GPT エラー: {e}")
        return text


# ==================================================================
# グローバルキー検出
# ==================================================================

# デバッグ: WHISPER_DEBUG_KEYS=1 でキーイベントをコンソールに表示
_DEBUG_KEYS = os.environ.get("WHISPER_DEBUG_KEYS", "").strip().lower() in ("1", "true", "yes")

# Alt 系ホットキーは keyboard ライブラリの release イベントが Windows で
# 取りこぼされるケースがあるため、Win32 GetAsyncKeyState で直接ポーリングする。
_ALT_POLL_INTERVAL = 0.025   # 40Hz


def _hotkey_poll_thread() -> None:
    """Win32 API ポーリングで Alt 系ホットキーの押下/離しを直接検出する。
    keyboard ライブラリのフックに依存しない。"""
    alt_was_pressed = False
    while True:
        alt_pressed = _is_hotkey_pressed()

        if alt_pressed and not alt_was_pressed:
            if _DEBUG_KEYS:
                vks = _get_hotkey_vks()
                states = {f"0x{vk:02X}": _is_vk_pressed(vk) for vk in vks}
                print(f"[Poll] Hotkey 押下検出 key={HOTKEY} VK={states}")
            if not _recording:
                threading.Thread(target=_do_start_recording, daemon=True).start()

        elif not alt_pressed and alt_was_pressed:
            if _DEBUG_KEYS:
                print(f"[Poll] Hotkey リリース検出 key={HOTKEY}")
            if _recording and not _rewrite_mode:
                threading.Thread(target=_do_stop_recording, daemon=True).start()

        alt_was_pressed = alt_pressed
        time.sleep(_ALT_POLL_INTERVAL)


def _setup_keyboard_hooks() -> None:
    """Right Ctrl のみ keyboard ライブラリで検出（Alt 系ホットキーはポーリング）"""
    def on_press(event: keyboard.KeyboardEvent) -> None:
        if event.name == "right ctrl" and not _recording:
            threading.Thread(target=_do_start_rewrite_recording, daemon=True).start()

    def on_release(event: keyboard.KeyboardEvent) -> None:
        if event.name == "right ctrl" and _recording and _rewrite_mode:
            threading.Thread(target=_do_stop_rewrite_recording, daemon=True).start()

    keyboard.on_press(on_press)
    keyboard.on_release(on_release)

    threading.Thread(target=_hotkey_poll_thread, daemon=True, name="alt-poll").start()
    print(f"[App] ホットキー検出開始 ({HOTKEY}=Win32ポーリング / Right Ctrl=キーフック)")


# ==================================================================
# Eel 安全呼び出しユーティリティ
# ==================================================================

def _safe_eel_call(func_name: str, *args) -> None:
    """WebSocket 未接続でも落ちないように try/except でラップ"""
    try:
        fn = getattr(eel, func_name)
        fn(*args)()
    except Exception:
        pass


# ==================================================================
# Eel 公開関数 (JS → Python)
# ==================================================================

@eel.expose
def on_navigate(page: str) -> dict:
    print(f"[Eel] ページ移動: {page}")
    return {"status": "ok", "page": page}


@eel.expose
def on_history_select(text: str) -> dict:
    print(f"[Eel] 履歴選択: {text[:60]}")
    return {"status": "ok"}


@eel.expose
def get_history() -> list[dict]:
    return _compact_history()


@eel.expose
def get_settings() -> dict:
    return _load_settings()


@eel.expose
def get_dictionary() -> list[dict]:
    return _load_dictionary()


@eel.expose
def add_dictionary_word(word: str, reading: str) -> list[dict]:
    entries = _load_dictionary()
    entries.append({"word": word, "reading": reading})
    _save_dictionary(entries)
    print(f"[App] 辞書追加: {word} ({reading})")
    return entries


@eel.expose
def delete_dictionary_word(word: str) -> list[dict]:
    entries = _load_dictionary()
    entries = [e for e in entries if e.get("word") != word]
    _save_dictionary(entries)
    print(f"[App] 辞書削除: {word}")
    return entries


@eel.expose
def get_snippets() -> list[dict]:
    return _load_snippets()


@eel.expose
def add_snippet(keyword: str, text: str) -> list[dict]:
    entries = _load_snippets()
    entries.append({"keyword": keyword, "text": text})
    _save_snippets(entries)
    print(f"[App] スニペット追加: {keyword}")
    return entries


@eel.expose
def delete_snippet(keyword: str) -> list[dict]:
    entries = _load_snippets()
    entries = [e for e in entries if e.get("keyword") != keyword]
    _save_snippets(entries)
    print(f"[App] スニペット削除: {keyword}")
    return entries


@eel.expose
def update_hotkey(key: str) -> dict:
    global HOTKEY
    settings = _load_settings()
    HOTKEY = _normalize_hotkey(key)
    settings["hotkey"] = HOTKEY
    _save_settings(settings)
    print(f"[App] ホットキー変更: {HOTKEY}")
    return {"status": "ok", "hotkey": HOTKEY}


@eel.expose
def update_language(language: str) -> dict:
    settings = _load_settings()
    normalized = _normalize_language(language)
    settings["language"] = normalized
    _save_settings(settings)
    print(f"[App] 音声認識言語を変更: {normalized}")
    return {"status": "ok", "language": normalized}


@eel.expose
def update_history_retention(days) -> dict:
    settings = _load_settings()
    normalized = _normalize_history_retention_days(days)
    settings["history_retention_days"] = normalized
    _save_settings(settings)
    compacted = _compact_history()
    label = "forever" if normalized is None else str(normalized)
    print(f"[App] 履歴保持期間を変更: {label}")
    return {"status": "ok", "history_retention_days": normalized, "history_count": len(compacted)}


# ==================================================================
# ウィンドウクローズコールバック
# ==================================================================

def _on_window_close(route: str, websockets: list) -> None:
    """ウィンドウを閉じてもアプリを終了しない（タスクトレイ常駐）"""
    global _eel_window_open
    _eel_window_open = False
    # 全 WebSocket が切れた場合のみログ
    if not websockets:
        print("[App] ウィンドウが閉じられました。タスクトレイで動作を継続します。")
        # 注意: sys.exit() は呼ばない → アプリ継続


# ==================================================================
# エントリポイント
# ==================================================================

def main() -> None:
    global HOTKEY
    if not _acquire_instance_lock():
        print("[App] 既にアプリが起動しています。二重起動を防止しました。")
        sys.exit(0)

    settings = _load_settings()
    HOTKEY = _normalize_hotkey(settings.get("hotkey"))
    _save_settings(settings)
    _compact_history()

    eel.init("web")

    print("[App] 音声入力ツールを起動しています...")
    print(f"[App] ホットキー: {HOTKEY}  (長押し → 録音、離す → ペースト)")
    if not _DEBUG_KEYS:
        print("[App] Alt キーが反応しない場合: 管理者で実行するか、WHISPER_DEBUG_KEYS=1 で起動してキー名を確認")

    # タスクトレイをバックグラウンドスレッドで起動
    tray_thread = threading.Thread(target=_start_tray, daemon=True)
    tray_thread.start()

    # グローバルキーボードフックを登録
    _setup_keyboard_hooks()

    # Eel ウィンドウを Chrome で起動（8888 が使用中なら別ポートを試す）
    port = 8888
    for _ in range(10):
        try:
            eel.start(
                "index.html",
                mode="chrome",
                size=WINDOW_SIZE,
                port=port,
                close_callback=_on_window_close,
                cmdline_args=[f"--user-data-dir={_CHROME_DATA_DIR}"],
            )
            break
        except OSError as e:
            in_use = (
                getattr(e, "winerror", None) == 10048
                or getattr(e, "errno", None) == errno.EADDRINUSE
                or "10048" in str(e)
                or "Address already in use" in str(e)
            )
            if in_use and port < 8898:
                port += 1
                print(f"[App] ポート {port - 1} 使用中のため {port} で起動します")
            else:
                raise


if __name__ == "__main__":
    main()
