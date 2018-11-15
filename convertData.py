import scipy.io.wavfile as wav
import librosa

def main():
	rate, sig = wav.read('./genres/blues/blues.00000.wav')
	print(rate, sig)
	print(sig.shape)
	print("hello")

	rate, sig = wav.read('./genres/blues/blues.00001.wav')
	print(rate, sig)
	print(sig.shape)
	print("hello2")

	audio, sr = librosa.core.load('./genres/blues/blues.00001.wav')
	print(audio, sr)
	print(audio.shape)


if __name__ == "__main__":
	main()