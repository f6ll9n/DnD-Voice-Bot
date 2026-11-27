"""Локальные тесты конфигурации, прокси LLM и голосов Piper — без Discord."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import LLMCfg, PersonaCfg, ROOT as CFG_ROOT, TTSCfg, load_config  # noqa: E402
from app import dave  # noqa: E402
from app.llm import LLM, is_local_url, is_low_quality_reply, strip_speaker_prefix  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402
from app.stt import STTCfg, Transcriber, _HALLUCINATIONS  # noqa: E402
from app.tts import SILERO_SPEAKERS, Speaker, rate_to_length_scale  # noqa: E402


class LocalConfigAndVoiceTests(unittest.TestCase):
    def test_is_local_url(self) -> None:
        self.assertTrue(is_local_url("http://127.0.0.1:11434/v1"))
        self.assertTrue(is_local_url("http://localhost:11434/v1"))
        self.assertTrue(is_local_url("http://[::1]:11434/v1"))
        self.assertFalse(is_local_url("https://api.groq.com/openai/v1"))
        self.assertFalse(is_local_url("http://192.168.1.50:11434/v1"))

    def test_llm_bypasses_proxy_only_for_local_urls(self) -> None:
        local_llm = LLM(LLMCfg(base_url="http://127.0.0.1:11434/v1", api_key="ollama", model="qwen2.5:3b"))
        remote_llm = LLM(LLMCfg(base_url="https://api.groq.com/openai/v1", api_key="test", model="llama"))
        try:
            self.assertIsNotNone(local_llm._http_client)
            self.assertIsNone(remote_llm._http_client)
        finally:
            asyncio.run(local_llm.aclose())
            asyncio.run(remote_llm.aclose())

    def test_rate_to_length_scale(self) -> None:
        self.assertIsNone(rate_to_length_scale("+0%"))
        self.assertIsNone(rate_to_length_scale(""))
        self.assertGreater(rate_to_length_scale("-10%"), 1.0)
        self.assertLess(rate_to_length_scale("+10%"), 1.0)

    def test_config_personas_use_distinct_valid_tts_voices(self) -> None:
        cfg = load_config()
        self.assertEqual(cfg.tts.engine, "silero")
        self.assertTrue((CFG_ROOT / cfg.tts.silero_model).is_file(), f"Не найдена модель: {cfg.tts.silero_model}")
        voices = [p.voice for p in cfg.personas.values()]
        self.assertEqual(len(voices), len(set(voices)), "У персонажей должны быть разные локальные голоса")
        for persona in cfg.personas.values():
            self.assertIn(persona.voice, SILERO_SPEAKERS)

    def test_silero_synthesizes_pcm_and_skips_empty_punctuation(self) -> None:
        speaker = Speaker(TTSCfg(engine="silero", silero_model="models/silero/v5_ru.pt"))
        pcm = speaker._silero_pcm("Привет, милок!", "eugene", strict=True)
        self.assertGreater(len(pcm), 10_000)
        self.assertEqual(len(pcm) % 4, 0)
        self.assertEqual(speaker._silero_pcm("...?!", "eugene"), b"")

    def test_silero_does_not_fallback_to_edge_on_missing_model(self) -> None:
        speaker = Speaker(TTSCfg(engine="silero", silero_model="models/silero/missing.pt"))
        self.assertEqual(speaker._silero_pcm("тест", "eugene"), b"")
        self.assertEqual(speaker.engine, "silero")

    def test_piper_does_not_fallback_to_edge_on_missing_model(self) -> None:
        speaker = Speaker(TTSCfg(engine="piper", piper_model="models/piper/missing.onnx"))
        pcm = speaker._piper_pcm("тест", "models/piper/missing.onnx")
        self.assertEqual(pcm, b"")
        self.assertEqual(speaker.engine, "piper")

    def test_dave_blocks_encrypted_packets_before_session_ready(self) -> None:
        class _Conn:
            dave_protocol_version = 1
            dave_session = type("_Sess", (), {"ready": False})()

        class _Sink:
            voice_client = type("_VC", (), {"_connection": _Conn()})()

        class _Decoder:
            ssrc = 42
            sink = _Sink()
            _cached_id = 100

        class _Packet:
            sequence = 1
            timestamp = 960
            decrypted_data = b"encrypted-payload"

        self.assertFalse(dave._apply_dave(_Decoder(), _Packet()))

    def test_dave_decode_packet_recovers_from_opus_error(self) -> None:
        from discord.opus import OpusError

        class _Conn:
            dave_protocol_version = 0
            dave_session = None

        class _Sink:
            voice_client = type("_VC", (), {"_connection": _Conn()})()

        class _FakeOpusError(OpusError):
            def __init__(self) -> None:
                Exception.__init__(self, "corrupted stream")

        class _OpusDec:
            def decode(self, data, fec=False):
                if data is not None:
                    raise _FakeOpusError()
                return b"silence"

        class _Decoder:
            ssrc = 77
            sink = _Sink()
            _cached_id = 100
            _decoder = _OpusDec()

        class _Packet:
            sequence = 5
            timestamp = 1920
            decrypted_data = b"corrupted"

            def __bool__(self):
                return True

        pkt, pcm = dave._decode_packet(_Decoder(), _Packet())
        self.assertIsNotNone(pkt)
        self.assertEqual(pcm, b"silence")

    def test_llm_guards_and_reset(self) -> None:
        persona = PersonaCfg(key="barman", title="Хома, бармен", file="", trigger_words=["бармен", "хома"])
        self.assertEqual(strip_speaker_prefix("Хома: Плати вперёд.", persona, "Фрикция"), "Плати вперёд.")
        self.assertEqual(strip_speaker_prefix("Фрикция: Дай пива", persona, "Фрикция"), "")
        self.assertTrue(is_low_quality_reply("你好世界"))
        self.assertTrue(is_low_quality_reply("Плати вперёд.", "Плати вперёд."))
        self.assertFalse(is_low_quality_reply("Держи кружку.", "Плати вперёд."))

        llm = LLM(LLMCfg(base_url="http://127.0.0.1:11434/v1", api_key="ollama", model="qwen2.5:3b"))
        try:
            llm._history["k1"].append({"role": "user", "content": "hi"})
            llm._memory["k1"] = "Фрикция заплатил 5 медяков"
            llm.reset("k1")
            self.assertEqual(len(llm._history["k1"]), 0)
            self.assertNotIn("k1", llm._memory)
        finally:
            asyncio.run(llm.aclose())

    def test_pipeline_followup_filter(self) -> None:
        self.assertTrue(Pipeline._looks_like_followup("5 медяков не много, может скинешь чуток?"))
        self.assertTrue(Pipeline._looks_like_followup("Держи пять медяков за пиво"))
        self.assertFalse(Pipeline._looks_like_followup("Ну да"))
        self.assertFalse(Pipeline._looks_like_followup("Он, я понял, спасибо"))

    def test_stt_skips_quiet_noise_and_detects_hallucinations(self) -> None:
        import numpy as np

        tr = Transcriber(STTCfg())
        tr._model = object()  # проверка срабатывает до обращения к модели
        quiet = np.full(48_000, 10, dtype=np.int16)
        self.assertEqual(tr.transcribe(quiet), "")
        self.assertIsNotNone(_HALLUCINATIONS.match("Субтитры субтитров А.Семкин"))
        self.assertIsNone(_HALLUCINATIONS.match("Хома, налей пива!"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
