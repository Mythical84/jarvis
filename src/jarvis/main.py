import pathlib
import time
import openwakeword
import pyaudio
import numpy as np
import openwakeword.utils
from openwakeword.model import Model
import os
import platform
import dotenv
import json
from groq import Groq
import mp3

dotenv.load_dotenv()

MODEL_PATH = 'models'
MIC_CHUNK_SIZE = 1280
MODEL_KEY=os.getenv('MODEL_KEY')

if not os.path.exists(MODEL_PATH): os.mkdir(MODEL_PATH)
openwakeword.utils.download_models(model_names=['jarvis'], target_directory=MODEL_PATH)

audio = pyaudio.PyAudio()
mic_stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=MIC_CHUNK_SIZE)

inference = 'onnx' if platform.system() == 'Windows' else 'tflite'
owwModel = Model(inference_framework=inference, wakeword_models=[f'{MODEL_PATH}/hey_jarvis_v0.1.{inference}'])

groq = Groq(api_key=MODEL_KEY)

cooldown = time.time()

while True:
	audio = np.frombuffer(mic_stream.read(MIC_CHUNK_SIZE), dtype=np.int16)
	prediction = owwModel.predict(audio)
	if prediction[next(iter(prediction))] > 0.5 and cooldown <= time.time():
		print('wakeword detected')
		audio_chunks = []
		current_time = time.time()
		while time.time() < current_time + 3:
			audio_chunks.append(np.frombuffer(mic_stream.read(MIC_CHUNK_SIZE), dtype=np.int16))
		audio_chunks = np.array(audio_chunks)
		raw_pcm = b''.join(audio_chunks)
		with open('output.mp3', 'wb') as file:
			encoder = mp3.Encoder(file)
			encoder.set_bit_rate(64)
			encoder.set_sample_rate(16000)
			encoder.set_channels(1)
			encoder.set_quality(2)
			encoder.set_mode(mp3.MODE_SINGLE_CHANNEL)
			encoder.write(raw_pcm)
			encoder.flush()

		text = groq.audio.transcriptions.create(
			file=pathlib.Path('output.mp3'),
			model='whisper-large-v3-turbo',
			response_format='verbose_json',
			language='en'
		)

		print(json.dumps(text, indent=2, default=str))
		cooldown = time.time() + 1
