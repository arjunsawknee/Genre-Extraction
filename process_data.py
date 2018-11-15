import os
import sox
import numpy as np
import librosa
import pandas as pd

def main():
	"""
		Converts all files of the GTZAN dataset
		to the WAV (uncompressed) format.
	"""
#	df2 = pd.read_csv('songs.csv')
#	print(df2.shape)
#	exit()

	tfm = sox.Transformer()
	songs = np.zeros((1000, 40000))
	labels = np.zeros((1000, 1))
	counter = 0

	for filename in os.listdir('./genres/blues'):
		if filename.endswith(".au"):
			tfm.build('./genres/blues/'+filename, './genres/blues/'+filename[:-2]+'wav')
			audio, sr = librosa.core.load('./genres/blues/'+filename[:-2]+'wav')
			songs[counter] = audio[:40000]
			labels[counter] = 0
			counter += 1
	
	for filename in os.listdir('./genres/classical'):
		if filename.endswith(".au"):
			tfm.build('./genres/classical/'+filename, './genres/classical/'+filename[:-2]+'wav')
			audio, sr = librosa.core.load('./genres/classical/'+filename[:-2]+'wav')
			songs[counter] = audio[:40000]
			labels[counter] = 1
			counter += 1

	for filename in os.listdir('./genres/country'):
		if filename.endswith(".au"):
			tfm.build('./genres/country/'+filename, './genres/country/'+filename[:-2]+'wav')
			audio, sr = librosa.core.load('./genres/country/'+filename[:-2]+'wav')
			songs[counter] = audio[:40000]
			labels[counter] = 2
			counter += 1

	for filename in os.listdir('./genres/disco'):
		if filename.endswith(".au"):
			tfm.build('./genres/disco/'+filename, './genres/disco/'+filename[:-2]+'wav')
			audio, sr = librosa.core.load('./genres/disco/'+filename[:-2]+'wav')
			songs[counter] = audio[:40000]
			labels[counter] = 3
			counter += 1

	for filename in os.listdir('./genres/hiphop'):
		if filename.endswith(".au"):
			tfm.build('./genres/hiphop/'+filename, './genres/hiphop/'+filename[:-2]+'wav')
			audio, sr = librosa.core.load('./genres/hiphop/'+filename[:-2]+'wav')
			songs[counter] = audio[:40000]
			labels[counter] = 4
			counter += 1

	for filename in os.listdir('./genres/jazz'):
		if filename.endswith(".au"):
			tfm.build('./genres/jazz/'+filename, './genres/jazz/'+filename[:-2]+'wav')
			audio, sr = librosa.core.load('./genres/jazz/'+filename[:-2]+'wav')
			songs[counter] = audio[:40000]
			labels[counter] = 5
			counter += 1

	for filename in os.listdir('./genres/metal'):
		if filename.endswith(".au"):
			tfm.build('./genres/metal/'+filename, './genres/metal/'+filename[:-2]+'wav')
			audio, sr = librosa.core.load('./genres/metal/'+filename[:-2]+'wav')
			songs[counter] = audio[:40000]
			labels[counter] = 6
			counter += 1

	for filename in os.listdir('./genres/pop'):
		if filename.endswith(".au"):
			tfm.build('./genres/pop/'+filename, './genres/pop/'+filename[:-2]+'wav')
			audio, sr = librosa.core.load('./genres/pop/'+filename[:-2]+'wav')
			songs[counter] = audio[:40000]
			labels[counter] = 7
			counter += 1

	for filename in os.listdir('./genres/reggae'):
		if filename.endswith(".au"):
			tfm.build('./genres/reggae/'+filename, './genres/reggae/'+filename[:-2]+'wav')
			audio, sr = librosa.core.load('./genres/reggae/'+filename[:-2]+'wav')
			songs[counter] = audio[:40000]
			labels[counter] = 8
			counter += 1

	for filename in os.listdir('./genres/rock'):
		if filename.endswith(".au"):
			tfm.build('./genres/rock/'+filename, './genres/rock/'+filename[:-2]+'wav')
			audio, sr = librosa.core.load('./genres/rock/'+filename[:-2]+'wav')
			songs[counter] = audio[:40000]
			labels[counter] = 9
			counter += 1

	songs = pd.DataFrame(songs)
	labels = pd.DataFrame(labels)
	songs.to_csv('songs.csv', index = False)
	labels.to_csv('labels.csv', index = False)

	print('Conversion done')

if __name__ == '__main__':
	main()
