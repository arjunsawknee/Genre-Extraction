import os
import sox
import numpy as np
import librosa

def main():
	"""
		Converts all files of the GTZAN dataset
		to the WAV (uncompressed) format.
	"""
	tfm = sox.Transformer()
	for filename in os.listdir('./genres/blues'):
		if filename.endswith(".au"):
			print(filename)
			tfm.build('./genres/blues/'+filename, './genres/blues/'+filename[:-2]+'wav')
	print(librosa.core.load('./genres/blues/blues.00000.wav', sr=None))
	print('Conversion done')

if __name__ == '__main__':
	main()
