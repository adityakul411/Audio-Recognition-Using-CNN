import wave
import time 
import ntpath
import numpy as np
import tflite_runtime.interpreter as tflite

file_path = "path/to/wav/file"
file = ntpath.basename(file_path)
real = file.rsplit('.', 1)[0]

ifile = wave.open(file_path)
samples = ifile.getnframes()
audio = ifile.readframes(samples)

audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
max_int16 = 2**15

audio_normalised = audio_as_np_float32 / max_int16
audio_normalised = np.ndarray.astype(audio_normalised, 'float32')

zero_padding = np.zeros(48000-np.shape(audio_normalised)[0])
equal_length = np.concatenate([audio_normalised, zero_padding], 0)

frame_length=255
frame_step=128

w = np.hanning(frame_length+1)
X = np.array([np.fft.rfft(w*equal_length[i:i+frame_length+1]) 
              for i in range(0, len(equal_length)-frame_length, frame_step)])

X = np.abs(X)

spectrogram = np.reshape(X, [1,374,129,1])
spectrogram = spectrogram.astype(np.float32)

begin = time.time()

interpreter = tflite.Interpreter(model_path="path/to/tflite/model")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], spectrogram)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

end = time.time()

fnl_out = output_data.flatten()

data = fnl_out*100

def index(i):
    switcher={
        0:'Backup',
        1:'Connection',
        2:'Device Diagnostic', 
        3:'Device Info',
        4:'Hello PDM',
        5:'No',
        6:'Reboot PDM',
        7:'Reboot to recovery mode',
        8:'Remove USB',
        9:'Run all',
        10:'Start local setup',
        11:'Stop local setup',
        12:'System setup',
        13:'Yes'
        
    }
    return switcher.get(i,"Invalid")

for i in range(len(data)):
    if(data[i]==max(data)):
        print("\nInput audio:", real)
        print("Prediction:", index(i))
        print("Confidence Score", data[i])

print(f"Total execution time of tflite network for inference is {end - begin}\n")