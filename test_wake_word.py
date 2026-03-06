import pvporcupine
from pvrecorder import PvRecorder

print("⏳ Loading Wake Word model...")

# 1. حطي الكود بتاعك واسم الملف هنا
ACCESS_KEY = "cG9a+pC+tp8RikfOgXbXNl3lpIMTpoEPG1bpfjAteGCe5geXBY69Jw==" 
KEYWORD_PATH = "E:\Downloads\sherpa-onnx-streaming-zipformer-en-20M-2023-02-17\sherpa-onnx-streaming-zipformer-en-20M-2023-02-17\Wheel-see_en_windows_v4_0_0.ppn" 

try:
    # 2. تهيئة الموديل
    porcupine = pvporcupine.create(
        access_key=ACCESS_KEY,
        keyword_paths=[KEYWORD_PATH]
    )
    
    # 3. تشغيل المايكروفون
    recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
    recorder.start()
    
    print("✅ System Ready!")
    print("😴 The wheelchair is SLEEPING. Say 'Wheel see' to wake it up...")
    print("-" * 50)

    # 4. الاستماع اللحظي
    while True:
        pcm = recorder.read()
        keyword_index = porcupine.process(pcm)
        
        if keyword_index >= 0:
            print("🚨 WAKE WORD DETECTED! The wheelchair is AWAKE! 🚨")
            
except KeyboardInterrupt:
    print("\n🛑 Stopping...")
finally:
    # تنظيف الميموري بعد القفل
    if 'recorder' in locals():
        recorder.delete()
    if 'porcupine' in locals():
        porcupine.delete()