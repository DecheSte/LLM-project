import os
import queue
import atexit
import random
import sys
import threading
import time
import tkinter as tk
import cv2
import numpy as np
import pygame
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ultralytics import YOLO

# ==========================================
# 初始化
# ==========================================
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from ultralytics import YOLO

if not torch.cuda.is_available():
    raise RuntimeError("未检测到 CUDA 环境！请确保安装了 GPU 版本的 PyTorch！")

print("[1/2] 正在加载 Qwen2.5-Coder (7B)")
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 配置 bitsandbytes 4-bit 动态量化锁
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

qwen_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map={"": "cuda:0"}
)

print("[2/2] 正在加载 YOLOv8 姿态估计引擎...")
yolo_model = YOLO("yolov8n-pose.pt")

print("三模态集群部署完毕")
task_queue = queue.Queue()

def cleanup_temp_audio_files():
    """不管程序如何退出（手动关闭/崩溃），全自动扫干擦净目录下的所有临时预警音频"""
    print("\n[系统正在安全关闭] 正在触发临终清理钩子，开始清扫临时音频残留...")
    
    # 匹配当前目录下所有 alert_*.wav 和 fallback_temp.mp3
    patterns = ["alert_*.wav", "fallback_temp.mp3", "alert_*.mp3"]
    
    removed_count = 0
    for pattern in patterns:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                removed_count += 1
            except Exception:
                pass # 如果文件正被别的进程死锁，跳过即可
                
    print(f"[清扫完毕] 已成功强力粉碎 {removed_count} 个残留临时垃圾音频！进程安全退出。")

# 向系统注册这个钩子
atexit.register(cleanup_temp_audio_files)

# ==========================================
# 辅助音频函数（修复 play_audio_pygame 与降级兜底未定义问题）
# ==========================================
def play_audio_pygame(audio_path):
    """使用 Pygame 播放生成的无损 WAV 音频流"""
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()

        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.unload()  # 播放完必须释放文件占用，防止文件写锁
        
        # 异步安全垃圾回收
        def auto_cleaner(file_to_del):
            time.sleep(2)
            try:
                if os.path.exists(file_to_del):
                    os.remove(file_to_del)
            except: pass
        threading.Thread(target=auto_cleaner, args=(audio_path,), daemon=True).start()
    except Exception as e:
        print(f"播放音频失败: {e}")

def run_edge_tts_fallback(text, output_file):
    """降级安全补偿锁：若高级克隆未开，秒级调用微软 Edge-TTS 读中文兜底"""
    try:
        voice_character = "zh-CN-XiaoxiaoNeural"
        print(f"正在通过 Edge-TTS 渲染本地降级声线...")
        clean_text = text.replace('"', '').replace("'", "")
        
        fallback_mp3 = "fallback_temp.mp3"
        tts_command = f'edge-tts --voice {voice_character} --text "{clean_text}" --write-media {fallback_mp3} --pitch=+12Hz --rate=+12%'
        os.system(tts_command)
        
        if os.path.exists(fallback_mp3):
            print("正在播放降级兜底预警...")
            play_audio_pygame(fallback_mp3)
    except Exception as e:
        print(f"降级语音生成或播放失败: {e}")


# ==========================================
# 2. 核心大模型与高度拟真音色合成逻辑
# ==========================================
def call_qwen_alert(seconds_slouched):
    messages = [
        {
            "role": "system", 
            "content": (
                "你是一个语气毒舌、傲娇的硬核动漫AI女仆。主人的电脑坐姿非常糟糕，现在坐在椅子上严重驼背了！\n"
                "请直接用【一句话纯正、地道的二次元日语】狠狠地训斥和警告主人，让他把背挺直！\n"
                "场景锁：主人当前是【坐在椅子上（椅子に座っている）】玩电脑。你的所有吐槽必须基于【坐姿】！\n"
                "铁律1：你的吐槽主题必须而且只能聚焦于【驼背、坐姿难看、把背挺直（背筋を伸ばす）、在椅子上坐正】！\n"
                "铁律2：严禁提及任何与‘站立（立ち上がる、立ち仕事）’、‘站着’相关的字眼！\n"
                "铁律3：控制在15个字以内的极其短促、有力的断句，绝对不准碎碎念说长句！\n"
                "只输出纯日语，不带任何中文和括号！"
            )
        },
        {"role": "user", "content": f"主人坐在椅子上已经严重驼背了。请立刻用短句日语训斥，让他坐正！"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(qwen_model.device)
    generated_ids = qwen_model.generate(
        **model_inputs, 
        max_new_tokens=35, 
        temperature=0.7,
        repetition_penalty=1.2,
        do_sample=True
    )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def call_qwen_translation(japanese_text):
    """【第二阶段】现场拦截日语并实时翻译为纯正中文"""
    messages = [
        {
            "role": "system", 
            "content": (
                "你是一个高水平的动漫语境翻译官。当前的场景是【硬核/毒舌AI正在警告和吐槽玩电脑严重驼背的主人】。\n"
                "请将输入的二次元日语精准翻译成中文。要求：\n"
                "1. 必须根据【驼背、坐姿异常】的场景进行意译，主语必须是【你/主人】，代表你在训斥或提醒用户坐直！\n"
                "2. 严禁出现字面直译导致的低级错误（例如把肩膀相关的词翻译成‘衣服脱落’或‘肩膀露出来’）。\n"
                "3. 语气要完美还原动漫里毒舌、傲娇的语境。只输出翻译后的纯中文结果，不要包含任何多余的解释。\n"
                "4. 场景基本固定在用户坐在椅子上，严禁出现‘站立’等翻译。"
            )
        },
        {"role": "user", "content": f"请翻译这句针对主人驼背的日语：{japanese_text}"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(qwen_model.device)
    generated_ids = qwen_model.generate(**model_inputs, max_new_tokens=64, temperature=0.2, do_sample=False) # 降低随机性确保翻译准确
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def process_ai_generation(elapsed_time):
    print("\n触发超时事件！")
    
    # 生成日语文本并储存
    voice_text = call_qwen_alert(elapsed_time)
    voice_text = voice_text.replace('"', '').replace('“', '').replace('”', '')
    print(f"🇯🇵 [一阶段·生成日语原文]: {voice_text}")
    
    # 进行翻译
    popup_text = call_qwen_translation(voice_text)
    popup_text = popup_text.replace('"', '').replace('“', '').replace('”', '')
    print(f"🇨🇳 [二阶段·汉化对齐]: {popup_text}")
    
    # 动态时间戳文件名
    timestamp = int(time.time())
    current_audio_file = f"alert_{timestamp}.wav"
    
    cloned_success = False
    headers = {"Content-Type": "application/json"}
    
    # 送入 GPT-SoVITS 渲染无损音频
    try:
        local_tts_url = "http://127.0.0.1:9880" 
        ref_wave_path = r"D:\GPT-SoVITS\GPT-SoVITS-v3lora-20250228\ref_voice.wav"
        if os.path.exists(ref_wave_path):
            print("正在向GPT-SoVITS发起最新版API克隆请求...")
            
            payload = {
                "ref_audio_path": ref_wave_path,
                "prompt_text": "",          
                "prompt_lang": "zh",        
                "text": voice_text,         
                "text_lang": "ja",          
                "text_split_method": "cut5", 
                "media_type": "wav"          
            }
            response = requests.post(f"{local_tts_url}/tts", json=payload, headers=headers, timeout=30)
            if response.status_code == 200 and len(response.content) > 1000:
                with open(current_audio_file, 'wb') as f:
                    f.write(response.content)
                print("专属音色渲染完毕！")
                cloned_success = True
    except Exception as e:
        print(f"本地服务器连接异常，原因: {e}")
        
    import threading # 如果脚本顶部没导，可以在这里局部导入

    # 异步播放调度
    if cloned_success and os.path.exists(current_audio_file):
        print("正在后台播放虚拟角色日语预警...")
        # 利用 Thread 让音频在后台播，代码直接往下走
        threading.Thread(target=play_audio_pygame, args=(current_audio_file,), daemon=True).start()
    else:
        print("语音克隆未成功，准备在后台启用 Edge-TTS 中文默认声线...")
        # 同样让 Edge-TTS 降级播放也走异步线程
        threading.Thread(target=run_edge_tts_fallback, args=(popup_text, current_audio_file), daemon=True).start()
        
    create_popup_window(popup_text)

def create_popup_window(warning_text):
    dialog = tk.Toplevel(root)
    dialog.title("智能坐姿管家高能预警")
    dialog.geometry("450x220+500+300")
    dialog.attributes('-topmost', True)
    
    label_title = tk.Label(dialog, text="监测到严重驼背！", font=("Helvetica", 14, "bold"), fg="red")
    label_title.pack(pady=10)
    label_content = tk.Label(dialog, text=warning_text, font=("Microsoft YaHei", 11), wraplength=400, justify="left")
    label_content.pack(pady=15, padx=20)
    btn_close = tk.Button(dialog, text="我知道了，立刻坐正", font=("Microsoft YaHei", 10), command=dialog.destroy)
    btn_close.pack(pady=10)

# ==========================================
# 3. 跨线程安全队列循环
# ==========================================
def check_queue():
    try:
        elapsed_time = task_queue.get_nowait()
        process_ai_generation(elapsed_time)
    except queue.Empty:
        pass
    root.after(100, check_queue)

def yolo_video_loop():
    cap = cv2.VideoCapture(0)
    slouch_start_time = None
    alert_triggered = False
    TIME_THRESHOLD = 5 

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        results = yolo_model(frame, stream=True, verbose=False)

        for r in results:
            if r.keypoints is not None and len(r.keypoints.data) > 0:
                keypoints = r.keypoints.data[0].cpu().numpy()
                try:
                    nose_y = keypoints[0][1]
                    avg_shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
                    status_text = "正常坐姿"

                    if (avg_shoulder_y - nose_y) < 120:
                        if slouch_start_time is None:
                            slouch_start_time = time.time()
                        else:
                            elapsed_time = time.time() - slouch_start_time
                            status_text = f"驼背中! ({int(elapsed_time)}s)"

                            if elapsed_time > TIME_THRESHOLD and not alert_triggered:
                                task_queue.put(elapsed_time)
                                alert_triggered = True
                    else:
                        slouch_start_time = None
                        if alert_triggered:
                            alert_triggered = False

                    color = (0, 255, 0) if "Good" in status_text else (0, 0, 255)
                    cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                except: pass
            cv2.imshow("Sitting Posture Agent System", r.plot())

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    print("系统正在安全关闭...")
    os._exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw() 
    root.after(100, check_queue)
    yolo_thread = threading.Thread(target=yolo_video_loop, daemon=True)
    yolo_thread.start()
    root.mainloop()