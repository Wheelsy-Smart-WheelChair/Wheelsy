import sherpa_onnx
import sounddevice as sd
import numpy as np
import time 

print("⏳ Loading the model, please wait...")

recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens="tokens.txt",
    encoder="encoder-epoch-99-avg-1.int8.onnx",
    decoder="decoder-epoch-99-avg-1.int8.onnx",
    joiner="joiner-epoch-99-avg-1.int8.onnx",
    num_threads=1,
    sample_rate=16000,
    feature_dim=80,
)

stream = recognizer.create_stream()

print("✅ Model is ready!")
print("🎙️ Microphone is on... Try saying commands!")
print("-" * 50)

def callback(indata, frames, time, status):
    if status:
        print(status)
    samples = indata.flatten().astype(np.float32)
    stream.accept_waveform(16000, samples)

last_text = ""
try:
    with sd.InputStream(channels=1, dtype="float32", samplerate=16000, callback=callback):
        while True:
         
            start_time = time.perf_counter()
            
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            
            text = recognizer.get_result(stream)
            
            if text and text != last_text:
               
                end_time = time.perf_counter()
                
                
                processing_time_ms = (end_time - start_time) * 1000 
                
                print(f"🤖 Command: {text} | ⏱️ Processing Time: {processing_time_ms:.2f} ms")
                last_text = text
                
except KeyboardInterrupt:
    print("\n🛑 Test stopped successfully.")