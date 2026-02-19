"""
whisper-recorder / transcribe.py
スペースキーを押している間だけマイクから録音し、
OpenAI Whisper API で文字起こしするスクリプト。

依存ライブラリ:
    pip install sounddevice numpy scipy openai keyboard python-dotenv
"""

import io
import os
import sys
import time
import threading

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

import keyboard  # グローバルキーフック（管理者権限不要、Windows/macOS/Linux対応）
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
load_dotenv()  # .env ファイルから OPENAI_API_KEY を読み込む

SAMPLE_RATE = 16000   # Hz（Whisper は 16kHz 推奨）
CHANNELS = 1          # モノラル
DTYPE = "int16"       # 16bit PCM
HOTKEY = "space"      # 録音トリガーキー
LANGUAGE = "ja"       # 文字起こし言語（None で自動判定）
MODEL = "whisper-1"   # Whisper モデル名

# ---------------------------------------------------------------------------
# OpenAI クライアント
# ---------------------------------------------------------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("[ERROR] OPENAI_API_KEY が設定されていません。")
    print("        .env ファイルに OPENAI_API_KEY=sk-... を記述するか、")
    print("        環境変数として設定してください。")
    sys.exit(1)

client = OpenAI(api_key=api_key)

# ---------------------------------------------------------------------------
# 録音状態管理
# ---------------------------------------------------------------------------
_recording = False
_audio_chunks: list[np.ndarray] = []
_lock = threading.Lock()
_stream: sd.InputStream | None = None


def _audio_callback(indata: np.ndarray, frames: int, time_info, status):
    """sounddevice のコールバック：録音中のみデータを蓄積する"""
    if status:
        print(f"[WARN] {status}", flush=True)
    with _lock:
        if _recording:
            _audio_chunks.append(indata.copy())


def start_recording():
    """録音開始"""
    global _recording, _audio_chunks, _stream
    with _lock:
        _recording = True
        _audio_chunks = []
    print("\n🎙  録音中... (スペースキーを離すと文字起こし開始)", flush=True)


def stop_and_transcribe():
    """録音停止 → Whisper API へ投げて結果を表示"""
    global _recording, _stream

    with _lock:
        _recording = False
        chunks = list(_audio_chunks)

    if not chunks:
        print("[INFO] 録音データがありません。\n", flush=True)
        return

    # numpy 配列を結合
    audio_np = np.concatenate(chunks, axis=0)

    duration = len(audio_np) / SAMPLE_RATE
    print(f"⏹  録音完了（{duration:.1f} 秒）。文字起こし中...", flush=True)

    # WAV としてメモリ上に書き出す
    wav_buffer = io.BytesIO()
    wav_write(wav_buffer, SAMPLE_RATE, audio_np)
    wav_buffer.seek(0)
    wav_buffer.name = "audio.wav"  # OpenAI SDK がファイル名でフォーマットを判定

    try:
        kwargs = dict(
            model=MODEL,
            file=wav_buffer,
        )
        if LANGUAGE:
            kwargs["language"] = LANGUAGE

        response = client.audio.transcriptions.create(**kwargs)
        text = response.text.strip()

        print("\n" + "─" * 50)
        print(f"📝 文字起こし結果:\n{text}")
        print("─" * 50 + "\n")

    except Exception as e:
        print(f"[ERROR] Whisper API 呼び出しに失敗しました: {e}\n", flush=True)


# ---------------------------------------------------------------------------
# キーイベントハンドラー
# ---------------------------------------------------------------------------
_space_pressed = False  # スペースの長押しリピートを防ぐ


def on_space_press(event):
    global _space_pressed
    if _space_pressed:
        return  # キーリピートを無視
    _space_pressed = True
    start_recording()


def on_space_release(event):
    global _space_pressed
    _space_pressed = False
    # 別スレッドで API 呼び出し（UI をブロックしない）
    threading.Thread(target=stop_and_transcribe, daemon=True).start()


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    print("=" * 50)
    print("  Whisper 音声文字起こしツール")
    print("=" * 50)
    print(f"  モデル  : {MODEL}")
    print(f"  言語    : {LANGUAGE or '自動'}")
    print(f"  録音キー: スペースキー（押している間だけ録音）")
    print("  終了    : Ctrl+C")
    print("=" * 50)
    print("\nスペースキーを押している間だけ録音します。準備ができたら押してください。\n")

    # sounddevice ストリームを開始（常時オープン、コールバックでデータ取捨）
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=_audio_callback,
    ):
        keyboard.on_press_key(HOTKEY, on_space_press)
        keyboard.on_release_key(HOTKEY, on_space_release)

        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[INFO] 終了します。")


if __name__ == "__main__":
    main()
