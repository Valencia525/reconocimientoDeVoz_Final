import sounddevice as sd
from scipy.io.wavfile import write

sd.default.samplerate = 16000
rec = sd.rec(16000, channels=1)
sd.wait()

write("prueba.wav", 16000, rec)
print("Guardado: prueba.wav")
