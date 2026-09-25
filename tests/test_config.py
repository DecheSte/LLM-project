import json
from pathlib import Path
import tempfile
import unittest
from posture_robot.config import load_config


class ConfigTests(unittest.TestCase):
    def test_example_and_invalid_values(self):
        example = Path(__file__).resolve().parents[1] / "config/settings.example.json"
        cfg = load_config(example)
        self.assertEqual(cfg["tts"]["base_url"], "http://127.0.0.1:9880")
        for section, key, value in [("posture", "duration", -1),
                                    ("posture", "confidence", 2),
                                    ("posture", "mode", "unknown"),
                                    ("qwen", "max_new_tokens", 1.5)]:
            changed = json.loads(example.read_text(encoding="utf-8"))
            changed[section][key] = value
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "settings.json"
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config(path)
