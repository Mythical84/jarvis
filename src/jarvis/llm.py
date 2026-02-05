import os
from groq import Groq 

class LLM():
	def __init__(self):
		self.groq = Groq(api_key=os.getenv('MODEL_KEY'))
		with open("src/jarvis/prompt.txt", "r") as file:
			self.system_prompt = "\n".join(file.readlines())

	def predict(self, json, memory):
		messages=[
			{
				"role": "system",
				"content": self.system_prompt
			},
		]

		# TODO: I could probably store the data already formatted in
		# the memory queue instead of formatting it here
		for mem in memory.get_short_term():
			messages.append({
				"role": "user",
				"content": mem['input']
			})

			messages.append({
				"role": "assistant",
				"content": mem['output']
			})

		messages.append({
				"role": "user",
				"content": json['input']
		})

		out = self.groq.chat.completions.create(
			model='llama-3.3-70b-versatile',
			messages=messages, # pyright: ignore
			temperature=1,
			top_p=1,
			stop=None
		)

		out = out.choices[0].message.content

		json['output'] = out

		memory.commit_short_term(json)

		return out
