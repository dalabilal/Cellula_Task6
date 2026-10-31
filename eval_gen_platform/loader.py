import os
import tempfile

import av
import numpy as np
from faster_whisper import WhisperModel

from config import WHISPER_MODEL

_model = None

LANGUAGES = {"english": "en", "arabic": "ar"}


def _get_model():
    global _model
    if _model is None:
        _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _model


def _decode_audio(path):
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    with av.open(path) as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):  # flush what is left
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def transcribe_file(audio_bytes, filename, language=None):
    if language:
        language = LANGUAGES.get(language.strip().lower(), language.strip().lower())
        if len(language) > 3:
            language = None  # not a valid code, let Whisper detect it

    ext = filename.split(".")[-1].lower() if "." in filename else "wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    try:
        audio = _decode_audio(path)
        if audio.size == 0:
            return "", language or "unknown"
        segments, info = _get_model().transcribe(audio, language=language)  # None = auto-detect
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text, info.language
    finally:
        os.remove(path)
