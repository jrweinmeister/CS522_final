# This works similar to the classify.py script, but makes makes the process seamless
# by recording and classifying at the same time.

import platform
import os
import numpy as np
import pickle
import pyaudio
import wave
from scipy import signal
from librosa import load as lib_load
from librosa import feature as lib_feature
import librosa

# Import sci-kit models
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


def load_data(path, sample_rate, file):
    # Create data lists
    samples = None
    labels = None
    classes = os.listdir(path)
    print('Loading data...')

    # for file in os.listdir(path):
    filename_path = os.path.join(path, file)
        # Load data
    samples, s = lib_load(filename_path, sr=sample_rate)
    # Append data and label
    # print('Loaded {}'.format(filename))

    return samples, classes


def extract_features(samples):
    window_numbers = 20  ## need to be determined
    domain_fv = []
    FFT_size = 1024

    # specs reshape: [frequency][time]
    #     print('Initial spectrogram shape: \t {}'.format(specs[i].shape))
    #     print('New specrogram length: \t {} \n'.format(specs_reshape[i].shape))

    # peak frequency

    # zero crossing rate
    # threshold crossing rate

    # MFCC
    mfccs = lib_feature.mfcc(y=samples, sr=sample_rate, n_mfcc=10,
                             win_length=int(np.ceil(FFT_size / window_numbers)),
                             hop_length=int(np.ceil(FFT_size / (2 * window_numbers))))
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # Spectral Centroid
    sc = lib_feature.spectral_centroid(y=samples, sr=sample_rate,
                                       win_length=int(np.ceil(FFT_size / window_numbers)),
                                       hop_length=int(np.ceil(FFT_size / (2 * window_numbers))))
    reshape_sc = []
    for x in sc:
        for j in x:
            reshape_sc.append(j)
    max_sc = max(reshape_sc)
    min_sc = min(reshape_sc)
    mean_sc = np.mean(reshape_sc)
    sc_fv = [max_sc, min_sc, mean_sc]

    # Bandwidth
    bw = lib_feature.spectral_bandwidth(y=samples, sr=sample_rate,
                                        win_length=int(np.ceil(FFT_size / window_numbers)),
                                        hop_length=int(np.ceil(FFT_size / (2 * window_numbers))))
    reshape_bw = []
    for x in bw:
        for j in x:
            reshape_bw.append(j)
    max_bw = max(reshape_bw)
    min_bw = min(reshape_bw)
    mean_bw = np.mean(reshape_bw)
    bw_fv = [max_bw, min_bw, mean_bw]

    # Combine all features for each sample
    all_fv = [mfccs_mean, sc_fv, bw_fv]
    reshape_fv = []
    for x in all_fv:
        for j in x:
            reshape_fv.append(j)

    # Get features for all samples
    domain_fv.append(reshape_fv)

    print('Features extracted...')

    return domain_fv


def classify(clf, domain_fv):
    print('Classifying...')
    # Specify which data to use, these are the only parameters that should change, the rest should remain the same.
    X = domain_fv

    # Convert X to numpy array if not imputing
    X = np.asarray(X)

    y_predict = clf.predict(X)

    return y_predict[0]


form_1 = pyaudio.paInt16 # 16-bit resolution
chans = 1 # 1 channel
samp_rate = 44100 # 44.1kHz sampling rate
chunk = 4096 # 2^12 samples for buffer
record_secs = 3 # seconds to record
dev_index = 1 # device index found by p.get_device_info_by_index(ii)
wav_output_filename = 'live_recording.wav' # name of .wav file

audio = pyaudio.PyAudio() # create pyaudio instantiation

# create pyaudio stream
stream = audio.open(format=form_1, rate=samp_rate, channels=chans,
                    input_device_index=dev_index, input=True,
                    frames_per_buffer=chunk)
print("recording")
frames = []

# Load model
pkl_filename = "pickle_model.pkl"

print('Loading model...')
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

class_history = np.full(10, fill_value="")

#Record and classify loop
try:
    while True: 
        # classifier loop
        if platform.system == 'Windows':
            os.system('cls')
        else:
            # for linux platfrom
            os.system('clear')

        print("recording in progress ... ")
        
        # loop through stream and append audio chunks to frame array
        for ii in range(0,int((samp_rate/chunk)*record_secs)):
            data = stream.read(chunk)
            frames.append(data)

        # print("finished recording")

        # save the audio frames as .wav file
        wavefile = wave.open(wav_output_filename,'wb')
        wavefile.setnchannels(chans)
        wavefile.setsampwidth(audio.get_sample_size(form_1))
        wavefile.setframerate(samp_rate)
        wavefile.writeframes(b''.join(frames))
        wavefile.close()

        # Get sample
        # Temporarily load data sample, replace with microphone
        audio_filename = 'live_recording.wav'
        root_path = os.getcwd()
        path = os.path.join(root_path, 'Data')
        sample_rate = 44100
        samples, classes = load_data(path, sample_rate, audio_filename)

        # Feature Extraction
        features = extract_features(samples)

        # Classify
        pred = classify(pickle_model, features)
        np.roll(class_history,-1)
        class_history[-1] = pred
        print("Predictions: \nnewest")
        hist_iter = 0
        for prediction in class_history:
            print(f"\tprediction {hist_iter}: {classes[prediction]}")
            hist_iter += 1
            # print(f"Predicted: {classes[pred]}")
            # print(f"Expected: {label}")
        print("oldest")

except KeyboardInterrupt:
    print("Stopping model")

# stop the stream, close it, and terminate the pyaudio instantiation
stream.stop_stream()
stream.close()
audio.terminate()
