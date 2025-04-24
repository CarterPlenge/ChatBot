import collections
import contextlib
import sys
import wave
import numpy as np
import sounddevice as sd
import webrtcvad
import os
import tempfile
from faster_whisper import WhisperModel

# === Settings ===
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION_MS = 30  # ms
VAD_MODE = 3  # 0=very sensitive, 3=very aggressive
MAX_SILENCE_FRAMES = 15  # how many silent frames before we stop
MAX_RECORD_SECONDS = 10  # prevent infinite recording

# init voice activated detection
vad = webrtcvad.Vad(VAD_MODE)

# Helper: convert float32 sounddevice data to 16-bit PCM
def float_to_pcm(audio):
    audio = np.clip(audio, -1, 1)
    return (audio * 32767).astype(np.int16)

# Helper: record and detect voice
def record_voice():
    print("Prepared to listen.")

    frame_size = int(SAMPLE_RATE * (FRAME_DURATION_MS / 1000.0))
    ring_buffer = collections.deque(maxlen=MAX_SILENCE_FRAMES)

    voiced_frames = []
    silence_count = 0
    triggered = False

    def callback(indata, frames, time, status):
        nonlocal silence_count, voiced_frames, triggered

        if status:
            print("status: ", status)

        pcm_audio = float_to_pcm(indata[:, 0])
        raw_data = pcm_audio.tobytes()

        is_speech = vad.is_speech(raw_data, SAMPLE_RATE)

        if not triggered:
            if is_speech:
                print("voice deteced, Recording...")
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
                silence_count = 0
            else:
                ring_buffer.append(raw_data)
                
        else:
            voiced_frames.append(raw_data)
            if not is_speech:
                silence_count += 1
            else:
                silence_count = 0

            if silence_count > MAX_SILENCE_FRAMES:
                raise sd.CallbackStop()  # stop stream

    with contextlib.closing(tempfile.NamedTemporaryFile(delete=False, suffix=".wav")) as f:
        filename = f.name

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32', callback=callback,
                        blocksize=frame_size):
        try:
            sd.sleep(MAX_RECORD_SECONDS * 1000)
        except sd.CallbackStop:
            pass

    # Save to .wav
    if voiced_frames:
        total_frames = len(b''.join(voiced_frames))
        duration_sec = total_frames / (SAMPLE_RATE * 2) # 2 bytes per sample
        if duration_sec < 1.0: 
            print("Seech too short to be valid.")
            os.remove(filename)
            return None
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(voiced_frames))
        print(f"Voice captured to: {filename}")
        return filename
    else:
        print("No speech detected.")
        os.remove(filename)
        return None

# Transcribe with Faster-Whisper
def transcribe(filename):
    model = WhisperModel("large-v3", compute_type="float16")
    segments, info = model.transcribe(filename)
    print("\nTranscription:")
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    return segment.text

if __name__ == "__main__":
    file = record_voice()
    if file:
        transcribe(file)
        os.remove(file)
