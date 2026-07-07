import sounddevice as sd
import numpy as np

def test_mic():
    samplerate = 16000
    chunk_duration = 0.1
    chunk_samples = int(samplerate * chunk_duration)
    print("Default device:", sd.default.device[0])
    try:
        stream = sd.InputStream(samplerate=samplerate, channels=1, blocksize=chunk_samples)
        stream.start()
        
        # Calibrate
        ambient_rms = []
        for _ in range(10): # 1 second
            indata, overflowed = stream.read(chunk_samples)
            rms = np.sqrt(np.mean(indata[:, 0]**2))
            ambient_rms.append(rms)
            
        base_rms = np.mean(ambient_rms)
        std_rms = np.std(ambient_rms)
        threshold = base_rms + (std_rms * 2) + 0.0005 # Small buffer
        print(f"Base RMS: {base_rms:.6f}, Std: {std_rms:.6f}, Suggested Threshold: {threshold:.6f}")
        
        stream.stop()
        stream.close()
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_mic()
