import os
import platform
import openwakeword.utils
from openwakeword.model import Model

class Wake():
	def __init__(self, model_path):
		self.model_path = model_path

		if not os.path.exists(self.model_path):
			os.mkdir(model_path)

		openwakeword.utils.download_models(
			target_directory=self.model_path
		)

		inference = 'onnx' if platform.system() == 'Windows' else 'tflite'
		self.model = Model(
			inference_framework=inference,
			wakeword_models=[f'{self.model_path}/hey_jarvis_v0.1.{inference}'],
			enable_speex_noise_suppression=True
		)

	def predict(self, audio):
		prediction = self.model.predict(audio)
		# pyright throws a false syntax error at the following line which gets ignored
		return prediction['hey_jarvis_v0.1'] # pyright: ignore
