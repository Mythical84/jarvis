import openwakeword
import pyaudio
import numpy as np
import openwakeword.utils
from openwakeword.model import Model
import os
import platform
import dotenv

dotenv.load_dotenv()

MODEL_PATH = 'models'
MIC_CHUNK_SIZE = 1280
MODEL_KEY=os.getenv('MODEL_KEY')

if not os.path.exists(MODEL_PATH): os.mkdir(MODEL_PATH)
openwakeword.utils.download_models(model_names=['jarvis'], target_directory=MODEL_PATH)

audio = pyaudio.PyAudio()
mic_stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=MIC_CHUNK_SIZE)

inference = 'onnx' if platform.system() == 'Windows' else 'tflite'
print(inference)
owwModel = Model(inference_framework=inference, wakeword_models=[f'{MODEL_PATH}/hey_jarvis_v0.1.{inference}'])

n_models = len(owwModel.models.keys())

os.system('clear')
print("\n\n")
print("#"*100)
print("Listening for wakewords...")
print("#"*100)
print("\n"*(n_models*3))

while True:
	audio = np.frombuffer(mic_stream.read(MIC_CHUNK_SIZE), dtype=np.int16)
	prediction = owwModel.predict(audio)

	n_spaces = 16
	output_string_header = """
		Model Name		   | Score | Wakeword Status
		--------------------------------------
		"""

	for mdl in owwModel.prediction_buffer.keys():
		scores = list(owwModel.prediction_buffer[mdl])
		curr_score = format(scores[-1], '.20f').replace("-", "")

		output_string_header += f"""{mdl}{" "*(n_spaces - len(mdl))}   | {curr_score[0:5]} | {"--"+" "*20 if scores[-1] <= 0.5 else "Wakeword Detected!"}
		"""

	print("\033[F"*(4*n_models+1))
	print(output_string_header, "							  ", end='\r')
