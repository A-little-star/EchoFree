import soundfile as sf
import librosa

wav, sr = sf.read("/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/ref.wav")
wav = librosa.resample(wav, orig_sr=48000, target_sr=16000)
sf.write("/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/ref_re.wav", wav, 16000)
