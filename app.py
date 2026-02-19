"""
app.py  –  Whisper Recorder  (Eel Desktop App)
======================================================
依存関係:
    pip install -r requirements.txt

起動:
    python app.py

動作:
    - Right Alt 長押し → 録音開始
    - Right Alt 離す  → Whisper で文字起こし → GPT-4o-mini で整形 → アクティブウィンドウへ自動ペースト
    - タスクトレイ常駐で動作継続
"""

from __future__ import annotations

import io
import os
import sys
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

SAMPLE_RATE   = 16_000   # Whisper 推奨
HOTKEY        = "right alt"
WINDOW_SIZE   = (1024, 680)

# ------------------------------------------------------------------
# アプリ状態
# ------------------------------------------------------------------
_recording     = False
_audio_frames: list[np.ndarray] = []
_tray_icon: pystray.Icon | None = None
_eel_window_open = False

# ------------------------------------------------------------------
# eel 初期化
# ------------------------------------------------------------------
eel.init("web")


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
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        while _recording:
            chunk, _ = stream.read(1024)
            _audio_frames.append(chunk.copy())


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
# 音声処理パイプライン: Whisper → GPT-4o-mini → ペースト
# ==================================================================

def _process_audio() -> None:
    if not _audio_frames:
        return

    # ① Whisper で文字起こし
    transcript = _transcribe()
    if not transcript:
        _safe_eel_call("js_show_notification", "❌ 文字起こしに失敗しました")
        return
    print(f"[Whisper] {transcript}")

    # ② GPT-4o-mini でテキスト整形
    formatted = _format_with_gpt(transcript)
    print(f"[GPT]     {formatted}")

    # ③ クリップボードにコピー → アクティブウィンドウへペースト
    pyperclip.copy(formatted)
    time.sleep(0.05)
    pyautogui.hotkey("ctrl", "v")

    # ④ UI に通知・履歴追加
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
        resp   = client.audio.transcriptions.create(
            model="whisper-1",
            file=buf,
            language="ja",
        )
        return resp.text.strip()
    except Exception as e:
        print(f"[App] Whisper エラー: {e}")
        return None


def _format_with_gpt(text: str) -> str:
    """GPT-4o-mini で音声認識テキストを自然な文章に整形"""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return text

    system_prompt = (
        "あなたは音声認識テキストの整形AIです。"
        "入力テキストを以下のルールで整形してください:\n"
        "1. 不自然な繰り返しや言い間違いを修正\n"
        "2. 句読点を適切に追加\n"
        "3. 自然な日本語または英語の文章にする\n"
        "4. 内容や意味は変えない\n"
        "5. 整形した文章だけを返す（説明・コメント不要）"
    )

    try:
        client = OpenAI(api_key=api_key)
        resp   = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": text},
            ],
            max_tokens=500,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[App] GPT エラー: {e}")
        return text       # フォールバック: そのまま返す


# ==================================================================
# グローバルキーボードフック
# ==================================================================

def _setup_keyboard_hooks() -> None:
    def on_press(event: keyboard.KeyboardEvent) -> None:
        if event.name in ("right alt", "alt gr") and not _recording:
            _do_start_recording()

    def on_release(event: keyboard.KeyboardEvent) -> None:
        if event.name in ("right alt", "alt gr") and _recording:
            _do_stop_recording()

    keyboard.on_press(on_press)
    keyboard.on_release(on_release)
    print(f"[App] グローバルキーフック設定完了 ({HOTKEY})")


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
def start_recording() -> None:
    """スペースキーなど JS から呼ぶ場合用（グローバルフックと併用可）"""
    _do_start_recording()


@eel.expose
def stop_recording() -> str:
    _do_stop_recording()
    return ""


@eel.expose
def update_hotkey(key: str) -> dict:
    """設定画面からホットキーを変更"""
    global HOTKEY
    HOTKEY = key
    print(f"[App] ホットキー変更: {key}")
    return {"status": "ok"}


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
    print("[App] 音声入力ツールを起動しています...")
    print(f"[App] ホットキー: {HOTKEY}  (長押し → 録音、離す → ペースト)")

    # タスクトレイをバックグラウンドスレッドで起動
    tray_thread = threading.Thread(target=_start_tray, daemon=True)
    tray_thread.start()

    # グローバルキーボードフックを登録
    _setup_keyboard_hooks()

    # Eel ウィンドウを Chrome で起動
    eel.start(
        "index.html",
        mode="chrome",
        size=WINDOW_SIZE,
        port=8888,
        close_callback=_on_window_close,
    )


if __name__ == "__main__":
    main()
