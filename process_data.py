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
	
	for filename in os.listdir('./genres/classical'):
		if filename.endswith(".au"):
			print(filename)
			tfm.build('./genres/classical/'+filename, './genres/classical/'+filename[:-2]+'wav')

	for filename in os.listdir('./genres/country'):
		if filename.endswith(".au"):
			print(filename)
			tfm.build('./genres/country/'+filename, './genres/country/'+filename[:-2]+'wav')

	for filename in os.listdir('./genres/disco'):
		if filename.endswith(".au"):
			print(filename)
			tfm.build('./genres/disco/'+filename, './genres/disco/'+filename[:-2]+'wav')

	for filename in os.listdir('./genres/hiphop'):
		if filename.endswith(".au"):
			print(filename)
			tfm.build('./genres/hiphop/'+filename, './genres/hiphop/'+filename[:-2]+'wav')

	for filename in os.listdir('./genres/jazz'):
		if filename.endswith(".au"):
			print(filename)
			tfm.build('./genres/jazz/'+filename, './genres/jazz/'+filename[:-2]+'wav')

	for filename in os.listdir('./genres/metal'):
		if filename.endswith(".au"):
			print(filename)
			tfm.build('./genres/metal/'+filename, './genres/metal/'+filename[:-2]+'wav')

	for filename in os.listdir('./genres/pop'):
		if filename.endswith(".au"):
			print(filename)
			tfm.build('./genres/pop/'+filename, './genres/pop/'+filename[:-2]+'wav')

	for filename in os.listdir('./genres/reggae'):
		if filename.endswith(".au"):
			print(filename)
			tfm.build('./genres/reggae/'+filename, './genres/reggae/'+filename[:-2]+'wav')

	for filename in os.listdir('./genres/rock'):
		if filename.endswith(".au"):
			print(filename)
			tfm.build('./genres/rock/'+filename, './genres/rock/'+filename[:-2]+'wav')

	print('Conversion done')

if __name__ == '__main__':
	main()
