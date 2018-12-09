import os
import sox
import numpy as np
import librosa
import pandas as pd
import numpy as np



def get_one_hot(label_num, num_classes = 10):
    one_hot = np.zeros((1,num_classes))
    one_hot[0, int(label_num)] = 1
    return one_hot

def main():
	"""
		Converts all files of the GTZAN dataset
		to the WAV (uncompressed) format.
	"""
#	df2 = pd.read_csv('songs.csv')
#	print(df2.shape)
#	exit()

	tfm = sox.Transformer()
	songs = np.zeros((10000, 40000))
	labels = np.zeros((10000, 1))
	onehotlabels = np.zeros((10000, 10))
	counter = 0

	allgenres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
	# Splits each song into 10 examples of shape (40000, 1) ~ 2 seconds each
	numsplit = 10
	sizesplit = 40000
	# generalized loop to process data
	for index in range(len(allgenres)):
		for filename in os.listdir('./genres/' + allgenres[index]):
			if filename.endswith(".au"):
				tfm.build('./genres/'+ allgenres[index] + '/' + filename, './genres/' + allgenres[index] + '/' + filename[:-2]+'wav')
				audio, sr = librosa.core.load('./genres/' + allgenres[index] + '/' + filename[:-2]+'wav')
				testmfcc = librosa.feature.mfcc(audio, sr)
				print(testmfcc)
				print(testmfcc.shape)
				print(audio.shape)
				exit()
				for j in range(numsplit):
					songs[counter] = audio[(sizesplit * j) : (sizesplit * (j + 1))]
					labels[counter] = index
					onehotlabels[counter] = get_one_hot(index)
					counter += 1

	'''for filename in os.listdir('./genres/blues'):
		if filename.endswith(".au"):
			tfm.build('./genres/blues/'+filename, './genres/blues/'+filename[:-2]+'wav')
			audio, sr = librosa.core.load('./genres/blues/'+filename[:-2]+'wav')
			print(audio.shape)
			exit()
			songs[counter] = audio[:40000]
			labels[counter] = 0
			onehotlabels[counter] = get_one_hot(0)
			counter += 1
	
	for filename in os.listdir('./genres/classical'):
		if filename.endswith(".au"):
			tfm.build('./genres/classical/'+filename, './genres/classical/'+filename[:-2]+'wav')
			audio, sr = librosa.core.load('./genres/classical/'+filename[:-2]+'wav')
			songs[counter] = audio[:40000]
			labels[counter] = 1
			onehotlabels[counter] = get_one_hot(1)
			counter += 1

	for filename in os.listdir('./genres/country'):
		if filename.endswith(".au"):
			tfm.build('./genres/country/'+filename, './genres/country/'+filename[:-2]+'wav')
			audio, sr = librosa.core.load('./genres/country/'+filename[:-2]+'wav')
			songs[counter] = audio[:40000]
			labels[counter] = 2
			onehotlabels[counter] = get_one_hot(2)
			counter += 1

	for filename in os.listdir('./genres/disco'):
		if filename.endswith(".au"):
			tfm.build('./genres/disco/'+filename, './genres/disco/'+filename[:-2]+'wav')
			audio, sr = librosa.core.load('./genres/disco/'+filename[:-2]+'wav')
			songs[counter] = audio[:40000]
			labels[counter] = 3
			onehotlabels[counter] = get_one_hot(3)
			counter += 1

	for filename in os.listdir('./genres/hiphop'):
		if filename.endswith(".au"):
			tfm.build('./genres/hiphop/'+filename, './genres/hiphop/'+filename[:-2]+'wav')
			audio, sr = librosa.core.load('./genres/hiphop/'+filename[:-2]+'wav')
			songs[counter] = audio[:40000]
			labels[counter] = 4
			onehotlabels[counter] = get_one_hot(4)
			counter += 1

	for filename in os.listdir('./genres/jazz'):
		if filename.endswith(".au"):
			tfm.build('./genres/jazz/'+filename, './genres/jazz/'+filename[:-2]+'wav')
			audio, sr = librosa.core.load('./genres/jazz/'+filename[:-2]+'wav')
			songs[counter] = audio[:40000]
			labels[counter] = 5
			onehotlabels[counter] = get_one_hot(5)
			counter += 1

	for filename in os.listdir('./genres/metal'):
		if filename.endswith(".au"):
			tfm.build('./genres/metal/'+filename, './genres/metal/'+filename[:-2]+'wav')
			audio, sr = librosa.core.load('./genres/metal/'+filename[:-2]+'wav')
			songs[counter] = audio[:40000]
			labels[counter] = 6
			onehotlabels[counter] = get_one_hot(6)
			counter += 1

	for filename in os.listdir('./genres/pop'):
		if filename.endswith(".au"):
			tfm.build('./genres/pop/'+filename, './genres/pop/'+filename[:-2]+'wav')
			audio, sr = librosa.core.load('./genres/pop/'+filename[:-2]+'wav')
			songs[counter] = audio[:40000]
			labels[counter] = 7
			onehotlabels[counter] = get_one_hot(7)
			counter += 1

	for filename in os.listdir('./genres/reggae'):
		if filename.endswith(".au"):
			tfm.build('./genres/reggae/'+filename, './genres/reggae/'+filename[:-2]+'wav')
			audio, sr = librosa.core.load('./genres/reggae/'+filename[:-2]+'wav')
			songs[counter] = audio[:40000]
			labels[counter] = 8
			onehotlabels[counter] = get_one_hot(8)
			counter += 1

	for filename in os.listdir('./genres/rock'):
		if filename.endswith(".au"):
			tfm.build('./genres/rock/'+filename, './genres/rock/'+filename[:-2]+'wav')
			audio, sr = librosa.core.load('./genres/rock/'+filename[:-2]+'wav')
			songs[counter] = audio[:40000]
			labels[counter] = 9
			onehotlabels[counter] = get_one_hot(9)
			counter += 1'''

	songs = pd.DataFrame(songs)
	labels = pd.DataFrame(labels)
	onehotlabels = pd.DataFrame(onehotlabels)
	songs.to_csv('songs.csv', index = False)
	labels.to_csv('labels.csv', index = False)
	onehotlabels.to_csv('onehotlabels.csv', index = False)

	print('Conversion done')

if __name__ == '__main__':
	main()
