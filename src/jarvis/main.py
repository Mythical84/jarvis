import time
import pyaudio
import numpy as np
import dotenv
import webrtcvad
import jarvis.transcription
import jarvis.llm
import jarvis.wake
import jarvis.memory
import jarvis.processor

# Constants
MODEL_PATH = '/home/tyler/projects/jarvis/models'
MIC_CHUNK_SIZE = 1024
SAMPLE_RATE = 16000 # Hz

def main():
	# Load api keys from .env
	dotenv.load_dotenv()

	# Initialize transcription, llm, and wake
	transcription = jarvis.transcription.Transcription(SAMPLE_RATE)
	llm = jarvis.llm.LLM()
	wake = jarvis.wake.Wake(MODEL_PATH)
	memory = jarvis.memory.Memory()
	processor = jarvis.processor.Processor()

	audio = pyaudio.PyAudio()
	mic_stream = audio.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=MIC_CHUNK_SIZE)
	# Speech recognition
	vad = webrtcvad.Vad()
	vad.set_mode(1)

	cooldown = time.time()

	print("Ready")

	while True:
		audio = np.frombuffer(mic_stream.read(MIC_CHUNK_SIZE), dtype=np.int16)
		prediction = wake.predict(audio) 
		if prediction > 0.5 and cooldown <= time.time():
			print('wakeword detected')
			raw_pcm = read_mic(mic_stream, vad)

			# If nothing besides the wake word is said, don't process the audio
			if len(raw_pcm) < 31000: 
				cooldown = time.time() + 1
				continue

			text = transcription.speech_to_text(raw_pcm)
			print(text)
			json = {"input": text}
			output = llm.predict(json, memory)
			print(output)

			processor.process_output(output)

			# Ensure a 1 second cooldown before the user can prompt again
			# This prevents a bug where the model continously reactivates
			cooldown = time.time() + 1

def read_mic(mic_stream, vad):
	audio_chunks = []
	grace_time = time.time() + 1
	data = mic_stream.read(320)
	while vad.is_speech(data, 16000) or time.time() < grace_time:
		audio_chunks.append(data)
		data = mic_stream.read(320)
	raw_pcm = b''.join(audio_chunks)

	return raw_pcm

if __name__ == "__main__":
	main()
