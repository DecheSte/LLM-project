import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
from posture_robot.services import Pipeline, SoVITS


def wav_bytes():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 100)
    return buf.getvalue()


class ServiceTests(unittest.TestCase):
    def test_order_and_saved_chinese_before_translation(self):
        with tempfile.TemporaryDirectory() as folder:
            events = []
            class FakeQwen:
                def complete(self, system, text):
                    if not events:
                        events.append("chinese")
                        return "快坐直！"
                    record = json.loads(next(Path(folder).glob("*.json")).read_text(encoding="utf-8"))
                    self_check = record["chinese"] == text == "快坐直！"
                    if not self_check:
                        raise AssertionError("翻译前必须保存同一句中文")
                    events.append("japanese")
                    return "背筋を伸ばして！"
            class FakeTTS:
                def synthesize(self, text):
                    events.append("audio")
                    return wav_bytes()
            alert = Pipeline(FakeQwen(), FakeTTS(), folder).prepare(5)
            self.assertEqual(events, ["chinese", "japanese", "audio"])
            self.assertEqual(alert.chinese, "快坐直！")

    def test_tts_failure_preserves_chinese_and_no_ready_alert(self):
        with tempfile.TemporaryDirectory() as folder:
            from unittest.mock import Mock
            qwen = Mock()
            qwen.complete.side_effect = ["坐直", "背筋を伸ばして"]
            tts = Mock()
            tts.synthesize.side_effect = TimeoutError("timeout")
            with self.assertRaises(TimeoutError):
                Pipeline(qwen, tts, folder).prepare(5)
            record = json.loads(next(Path(folder).glob("*.json")).read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["chinese"], "坐直")

    def test_tts_contract_and_invalid_audio(self):
        cfg = {"base_url": "http://127.0.0.1:9880", "ref_audio_path": "voice.wav",
               "prompt_text": "参考文本", "prompt_lang": "zh", "timeout": 120}
        with patch("posture_robot.services.post", return_value=wav_bytes()) as request:
            SoVITS(cfg).synthesize("背筋を伸ばして")
            args = request.call_args.args
            self.assertEqual(args[0], "http://127.0.0.1:9880/tts")
            self.assertEqual(args[1]["text_lang"], "ja")
            self.assertFalse(args[1]["streaming_mode"])
        for invalid in (b'{"error":"failure"}', wav_bytes()[:-3]):
            with patch("posture_robot.services.post", return_value=invalid):
                with self.assertRaises((wave.Error, ValueError)):
                    SoVITS(cfg).synthesize("test")


if __name__ == "__main__":
    unittest.main()
