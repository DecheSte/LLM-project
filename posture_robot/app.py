import argparse
import io
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time

from .config import load_config
from .core import PostureTimer, posture_distance
from .services import Pipeline, Qwen, SoVITS


def present_alert(root, alert):
    """先解码音频并构建隐藏弹窗，在同一个 GUI 回调里显示和播放。"""
    import tkinter as tk
    import pygame
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    sound = pygame.mixer.Sound(file=io.BytesIO(alert.audio))
    channel = pygame.mixer.find_channel(force=False)
    if channel is None:
        raise RuntimeError("没有可用音频播放通道")
    dialog = tk.Toplevel(root)
    dialog.withdraw()
    dialog.title("坐姿提醒")
    dialog.attributes("-topmost", True)
    dialog.resizable(False, False)
    tk.Label(dialog, text="坐直一点，让肩颈休息一下", font=("Microsoft YaHei", 15, "bold")).pack(padx=28, pady=18)
    tk.Label(dialog, text=alert.chinese, wraplength=420, font=("Microsoft YaHei", 12)).pack(padx=28, pady=10)
    def acknowledge():
        channel.stop()
        dialog.destroy()
    tk.Button(dialog, text="确定，我会坐正", command=acknowledge, width=22).pack(pady=20)
    # 必须点确定；语音结束不自动关闭，右上角关闭键也不绕过确认。
    dialog.protocol("WM_DELETE_WINDOW", lambda: None)
    dialog.update_idletasks()
    try:
        dialog.deiconify()
        dialog.lift()
        channel.play(sound)
        dialog.grab_set()
        root.wait_window(dialog)
    finally:
        channel.stop()
        if dialog.winfo_exists():
            dialog.destroy()


class Application:
    def __init__(self, root, cfg):
        import tkinter as tk
        import cv2
        from ultralytics import YOLO
        self.root, self.cfg, self.cv2 = root, cfg, cv2
        self.closed = False
        self.cap = None
        self.error_until = 0.0
        self.error_text = ""
        root.title("坐姿纠正机器人")
        self.status = tk.StringVar(value="正在加载模型……")
        tk.Label(root, textvariable=self.status, wraplength=650, font=("Microsoft YaHei", 11)).pack(padx=12, pady=12)
        self.preview = tk.Label(root)
        self.preview.pack()
        tk.Button(root, text="退出", command=self.close).pack(pady=10)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.update_idletasks()
        self.model = YOLO(cfg["camera"]["model"])
        self.pipeline = Pipeline(Qwen(cfg["qwen"]), SoVITS(cfg["tts"]), cfg["runtime"]["history_dir"])
        self.timer = PostureTimer(cfg["posture"]["threshold"], cfg["posture"]["duration"],
            cfg["posture"]["cooldown"], cfg["posture"]["max_gap"])
        self.cap = cv2.VideoCapture(cfg["camera"]["index"])
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError("无法打开摄像头，请检查 camera.index 和摄像头权限")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
        root.after(10, self.tick)

    def tick(self):
        if self.closed:
            return
        from PIL import Image, ImageTk
        try:
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError("摄像头画面读取失败，请退出后重新连接摄像头")
            result = self.model(frame, verbose=False, device=self.cfg["camera"]["device"])[0]
            points = result.keypoints
            p = self.cfg["posture"]
            distance = None
            if points is not None and len(points.data) == 1:
                distance = posture_distance(points.data[0].cpu().tolist(), p["confidence"], p["mode"] == "normalized")
            now = time.monotonic()
            trigger, elapsed = self.timer.update(distance, now)
            if now < self.error_until:
                text = self.error_text
            elif distance is None:
                text = "请保持单人入镜，露出鼻子和双肩；计时已清零"
            elif now < self.timer.eligible_at:
                text = f"冷却中，还剩 {self.timer.eligible_at - now:.0f} 秒 | 距离 {distance:.2f}"
            else:
                text = f"距离 {distance:.2f} / 阈值 {p['threshold']} | " + (
                    f"坐姿异常 {elapsed:.1f}/{p['duration']} 秒" if distance < p["threshold"] else "坐姿正常")
            self.status.set(text)
            rendered = self.cv2.cvtColor(result.plot(), self.cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rendered)
            image.thumbnail((800, 600))
            self.image = ImageTk.PhotoImage(image)
            self.preview.configure(image=self.image)
            if trigger:
                self.status.set("正在生成中文、翻译日语并合成语音，请稍候……")
                self.root.update_idletasks()
                try:
                    alert = self.pipeline.prepare(elapsed)
                    present_alert(self.root, alert)
                    self.timer.reset(time.monotonic())
                except Exception as exc:
                    logging.exception("提醒生成或播放失败")
                    delay = self.cfg["runtime"]["retry_delay"]
                    self.error_text = f"提醒失败：{exc}；{delay}秒后重新检测"
                    self.error_until = time.monotonic() + delay
                    self.status.set(self.error_text)
                    self.timer.reset(time.monotonic(), delay)
        except Exception as exc:
            logging.exception("检测停止")
            self.status.set(str(exc))
            self.cap.release()
            return
        if not self.closed:
            self.root.after(30, self.tick)

    def close(self):
        self.closed = True
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 + Qwen7B + GPT-SoVITS 坐姿机器人")
    parser.add_argument("--config", default="config/settings.json")
    parser.add_argument("--check-config", action="store_true", help="仅检查配置，不加载模型")
    parser.add_argument("--test-alert", action="store_true", help="跳过摄像头，验证真实文字和音频链路")
    args = parser.parse_args()
    root = app = None
    try:
        cfg = load_config(args.config)
        if args.check_config:
            print("配置结构检查通过（不代表模型、音色或服务已就绪）")
            return
        if not cfg["tts"]["ref_audio_path"]:
            raise ValueError("请先在配置中填写 tts.ref_audio_path")
        Path("runtime").mkdir(exist_ok=True)
        logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
            RotatingFileHandler("runtime/app.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")])
        import tkinter as tk
        root = tk.Tk()
        if args.test_alert:
            root.withdraw()
            pipeline = Pipeline(Qwen(cfg["qwen"]), SoVITS(cfg["tts"]), cfg["runtime"]["history_dir"])
            present_alert(root, pipeline.prepare(5))
        else:
            app = Application(root, cfg)
            root.mainloop()
    except Exception as exc:
        logging.exception("运行失败")
        parser.exit(1, f"错误：{exc}\n")
    finally:
        if app is not None and app.cap is not None:
            app.cap.release()
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass
        import sys
        if "pygame" in sys.modules:
            sys.modules["pygame"].mixer.quit()
