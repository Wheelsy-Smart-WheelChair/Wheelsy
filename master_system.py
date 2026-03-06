import pvporcupine
from pvrecorder import PvRecorder
import sherpa_onnx
import numpy as np
import time

print("⏳ Initializing the Smart Wheelchair System...")

# ==========================================
# Helper Function: Intent Extraction (Command Filter)
# ==========================================
def extract_command(text):
    """
    Analyzes the recognized text and extracts a single character command
    to be sent to the ESP32 via Bluetooth.
    """
    text = text.lower()
    
    if any(word in text for word in ["forward", "front", "straight", "ahead"]):
        return "F" # Forward
        
    elif any(word in text for word in ["back", "backward", "reverse", "behind"]):
        return "B" # Backward
        
    elif any(word in text for word in ["right", "write", "rite"]):
        return "R" # Right
        
    elif any(word in text for word in ["left", "lift"]):
        return "L" # Left
        
    elif any(word in text for word in ["stop", "halt", "wait", "stay"]):
        return "S" # Stop
        
    return None # Ignore irrelevant speech

# ==========================================
# 1. Setup Porcupine (Wake Word Model)
# ==========================================
ACCESS_KEY = "cG9a+pC+tp8RikfOgXbXNl3lpIMTpoEPG1bpfjAteGCe5geXBY69Jw==" 
KEYWORD_PATH = r"E:\Downloads\sherpa-onnx-streaming-zipformer-en-20M-2023-02-17\sherpa-onnx-streaming-zipformer-en-20M-2023-02-17\Wheel-see_en_windows_v4_0_0.ppn" 

porcupine = pvporcupine.create(
    access_key=ACCESS_KEY,
    keyword_paths=[KEYWORD_PATH]
)

# ==========================================
# 2. Setup Sherpa-ONNX (Command Recognition Model)
# ==========================================
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens="tokens.txt",
    encoder="encoder-epoch-99-avg-1.onnx",
    decoder="decoder-epoch-99-avg-1.onnx",
    joiner="joiner-epoch-99-avg-1.onnx",
    num_threads=1,
    sample_rate=16000,
    feature_dim=80,
)

# ==========================================
# 3. Setup Microphone & System Logic
# ==========================================
recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
recorder.start()

is_awake = False
awake_time = 0
LISTEN_DURATION = 5  # Listening window in seconds

print("✅ System Ready!")
print("😴 Wheelchair is SLEEPING. Say 'Wheelsy' to wake it up...")
print("-" * 50)

last_text = ""

try:
    while True:
        pcm = recorder.read()
        
        if not is_awake:
            # --- Sleep State: Listening only for Wake Word ---
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print("\n🚨 WAKE WORD DETECTED! Listening...")
                is_awake = True
                awake_time = time.time()
                sherpa_stream = recognizer.create_stream() 
                last_text = ""
                
        else:
            # --- Awake State: Listening for Commands ---
            samples = np.array(pcm, dtype=np.float32) / 32768.0
            sherpa_stream.accept_waveform(16000, samples)
            
            while recognizer.is_ready(sherpa_stream):
                recognizer.decode_stream(sherpa_stream)
                
            text = recognizer.get_result(sherpa_stream)
            
            # Print only if text changed and it contains a valid command
            if text and text != last_text:
                final_action = extract_command(text)
                
                if final_action:
                    # Clear line and print the final action and what was heard
                    print(f"\r🚀 Action: [{final_action}] (Heard: {text})" + " " * 15, end="", flush=True)
                
                last_text = text
                
            # Check if listening duration has passed
            if time.time() - awake_time > LISTEN_DURATION:
                print("\n😴 Timeout. Going back to sleep...")
                is_awake = False

except KeyboardInterrupt:
    print("\n🛑 System shutting down safely...")
finally:
    # Free up memory resources
    if 'recorder' in locals():
        recorder.delete()
    if 'porcupine' in locals():
        porcupine.delete()