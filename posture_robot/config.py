import json
import math
from pathlib import Path


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    for section in ("camera", "posture", "qwen", "tts", "runtime"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"缺少配置节: {section}")
    for section, names in {
        "posture": ["threshold", "duration", "cooldown", "max_gap"],
        "qwen": ["timeout", "max_new_tokens"], "tts": ["timeout"],
        "runtime": ["retry_delay"], "camera": ["width", "height"],
    }.items():
        for name in names:
            value = cfg[section][name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{section}.{name} 必须为正数")
    if cfg["posture"]["mode"] not in ("pixels", "normalized"):
        raise ValueError("posture.mode 仅支持 pixels / normalized")
    if not 0 < cfg["posture"]["confidence"] <= 1:
        raise ValueError("关键点置信度必须在 (0, 1] 内")
    if cfg["qwen"]["backend"] not in ("transformers", "openai"):
        raise ValueError("qwen.backend 仅支持 transformers / openai")
    for section, key in (("camera", "width"), ("camera", "height"), ("qwen", "max_new_tokens")):
        if not isinstance(cfg[section][key], int):
            raise ValueError(f"{section}.{key} 必须为整数")
    return cfg
