import os
import time
import sounddevice as sd
from scipy.io.wavfile import write

def record_voice(filename="output.wav", duration=5, sample_rate=22050):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    write(filename, sample_rate, audio)
    print(f"Saved recording to: {filename}")


if __name__ == "__main__":
    voice_name = input("Enter a name for the voice: ").strip()
    folder = os.path.join("voices", voice_name)
    os.makedirs(folder, exist_ok=True)

    num_clips = 5
    duration = 8  # seconds
    pause_time = 3  # seconds between recordings

    print(f"\nStarting recording for '{voice_name}' with {num_clips} clips.\n")

    for i in range(1, num_clips + 1):
        input(f"🎤 Press Enter to start recording sample {i}/{num_clips}...")
        filename = os.path.join(folder, f"{i}.wav")
        record_voice(filename, duration=duration)

        if i != num_clips:
            print(f"⏳ Pausing {pause_time} seconds before next sample...\n")
            time.sleep(pause_time)

    print("\n All samples recorded! You can now use this folder with Tortoise.")