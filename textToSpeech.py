import os
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice, load_voices
from pydub import AudioSegment
from pydub.playback import play

def play_wav(path):
    sound = AudioSegment.from_wav(path)
    play(sound)

class TTS:
    def __init__(self, voiceFolder:str="defaultVoice"):
        self.model = TextToSpeech()
        self.voice_samples, self.conditioning_latents = load_voice("voices/"+voiceFolder)


    def say(self, message):
        gen = self.model.tts_with_preset(
            text=message,
            voice_samples=self.voice_samples,
            conditioning_latents=self.conditioning_latents,
            preset="high_quality"
        )
        out_path = "output.wav"
        self.model.save_audio(out_path, gen)
        play_wav(out_path)
        os.remove(out_path) # clean up file
