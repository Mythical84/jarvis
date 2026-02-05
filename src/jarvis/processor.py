import subprocess

class Processor():
	def process_output(self, out):
		text = out.split("\n")
		match text[0]:
			case "capability 1:":
				ret = subprocess.run(';'.join(text[1:]), capture_output=True, shell=True)
				print(ret.stdout.decode())
