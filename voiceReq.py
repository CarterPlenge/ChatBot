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
SAMPLE_RATE = 16000         # Audio sample rate in Hz. higher numbers = higher quality = more computational expenses. #16kHZ is standard for speech rec; kindda like resolution
CHANNELS = 1                # Audio channels; 1 = mono, 2 = stereo; idk just leave it as one
FRAME_DURATION_MS = 30     # Higher = more data perframe = increase accuracy; WebRTC VAD only supports 10, 20 and 30
VAD_MODE = 3                # 0-3; higher values ignore background noise better but may miss quiet speech;
MAX_SILENCE_FRAMES = 15    # Stop recording after this many silent frames; silence_frames(15) * frame_duration(30ms) = 450ms of silence to stop
MAX_RECORD_SECONDS = 10    # max length of recording

# Initialize voice activity detector with specified aggressiveness
vad = webrtcvad.Vad(VAD_MODE)

# === Helper Functions ===

# Convert float32 audio samples (-1.0 to 1.0) to 16-bit PCM for VAD and saving to WAV
def float_to_pcm(audio):
    audio = np.clip(audio, -1, 1)  # Ensure values stay in range [-1, 1]
    return (audio * 32767).astype(np.int16)

# Record user's voice using microphone and WebRTC VAD
def record_voice():
    print("Prepared to listen.")

    frame_size = int(SAMPLE_RATE * (FRAME_DURATION_MS / 1000.0))  # Number of samples per frame. sample rate * frame duration in microseconds?
    ring_buffer = collections.deque(maxlen=MAX_SILENCE_FRAMES)    # Buffer to hold recent frames before speech starts. helps ensure we clearly capture the first word

    voiced_frames = []     # Store all voice-detected frames
    silence_count = 0      # Counter for how many silent frames in a row
    triggered = False

    # Called continuously with audio input
    def callback(indata, frames, time, status):
        nonlocal silence_count, voiced_frames, triggered

        if status:
            print("voiceReq.py - status: ", status)  # Print errors or warnings from sounddevice

        # Convert from float32 to PCM bytes to be interpreted
        pcm_audio = float_to_pcm(indata[:, 0])
        raw_data = pcm_audio.tobytes()

        # Use VAD to detect if this frame contains speech
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

    # Create temporary WAV file to hold the result
    with contextlib.closing(tempfile.NamedTemporaryFile(delete=False, suffix=".wav")) as f:
        filename = f.name

    # Start recording from microphone
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32', callback=callback,
                        blocksize=frame_size):
        try:
            sd.sleep(MAX_RECORD_SECONDS * 1000)  # Wait up to MAX_RECORD_SECONDS or until stopped
        except sd.CallbackStop:
            pass

    # If speech was detected, write the audio to a WAV file
    if voiced_frames:
        total_frames = len(b''.join(voiced_frames))
        duration_sec = total_frames / (SAMPLE_RATE * 2) # 2 bytes per sample
        if duration_sec < 1.0: 
            print("Seech too short to be valid.")
            os.remove(filename)
            return None
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit samples = 2 bytes
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(voiced_frames))  # Save all recorded audio
        print(f"Voice captured to: {filename}")
        return filename
    else:
        # If no voice was detected, delete the temporary file
        print("No speech detected.")
        os.remove(filename)
        return None

# Transcribe audio file using Faster-Whisper
def transcribe(filename):
    model = WhisperModel("large-v3", compute_type="float16")  # Load Whisper model (large version)
    segments, info = model.transcribe(filename)  # Perform transcription

    print("\nTranscription:")
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")  # Print each segment's timing and text

    # This currently only returns the *last* segment's text — not the full result!
    return segment.text

# === Main Program ===
if __name__ == "__main__":
    file = record_voice()     # Record voice from microphone
    if file:
        transcribe(file)      # Run transcription on saved WAV file
        os.remove(file)       # Clean up temporary file
