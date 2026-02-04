import os
import pathlib
from groq import Groq
import mp3

class Transcription():
	def __init__(self, sample_rate):
		self.groq = Groq(api_key=os.getenv('MODEL_KEY'))
		self.sample_rate = sample_rate

	def speech_to_text(self, raw_pcm):
		with open('output.mp3', 'wb') as file:
			encoder = mp3.Encoder(file)
			encoder.set_bit_rate(32)
			encoder.set_sample_rate(self.sample_rate)
			encoder.set_channels(1)
			encoder.set_quality(2) # 2-highest quality, 7-fastest
			encoder.set_mode(mp3.MODE_SINGLE_CHANNEL)
			encoder.write(raw_pcm)
			encoder.flush()

		text = self.groq.audio.transcriptions.create(
			file=pathlib.Path('output.mp3'),
			model='whisper-large-v3-turbo',
			response_format='verbose_json',
			language='en'
		)

		os.remove('output.mp3')

		return text.text.strip()
