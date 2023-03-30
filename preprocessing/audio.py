import librosa
import numpy as np
import os


""" 
    Preprocessing Audio Data:
    1. Load file audio (.wav) with limit duration, sample_rate and mono.
    2. Pad audio array samples.
    3. Extract log spectrogram from signal audio
    4. Normalise data (Canneled in TTS)
"""


class Loader:
    def __init__(self, sample_rate: int, duration: float, mono: bool = True):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load_data(self, file_path: str):
        signal, _ = librosa.load(file_path, sr=self.sample_rate, mono=self.mono, duration=self.duration)
        return signal

class Padder:
    def __init__(self, mode: str = "contant"):
        self.mode = mode
    def is_pad(self, signal, samples):
        if len(signal) < samples:
            return True
        return False

    def right_pad(self, signal, num):
        return np.pad(signal, (0, num), mode=self.mode)
    def left_pad(self, signal, num):
        return np.pad(signal, (num, 0), mode=self.mode)

    def pad(self, signal, samples):
        if self.is_pad(signal, samples):
            signal = self.right_pad(signal, samples - len(signal))
        return signal

class Extractor:
    def __init__(self, sample_rate: int, frame_size: int, hop_length: int, n_mels: int):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.n_mels = n_mels

    def extract(self, signal):
        log_mels = librosa.feature.melspectrogram(y=signal, sr=self.sample_rate, n_fft=self.frame_size, hop_length=self.hop_length, n_mels=self.n_mels)
        log_spectrogram = librosa.power_to_db(log_mels, ref=np.max)
        return log_spectrogram



class AudioProcessor:
    def __init__(self, sample_rate: int, duration: float, mono: bool, frame_size: int, hop_length: int, n_mels: int, mode: str = "constant", min: float = -1, max: float = 1):
        self.sample_rate = sample_rate
        if sample_rate is None:
            self.sample_rate = 22050
        self.duration = duration
        self.mono = mono
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mode = mode
        self.min = min
        self.max = max
        self.loader = Loader(sample_rate, duration, mono)
        self.padder = Padder(mode)
        self.extractor = Extractor(sample_rate, frame_size, hop_length, n_mels)
    def __process(self, file_path: str) -> np.ndarray:
        if file_path is None:
            return
        signal = self.loader.load_data(file_path)
        signal = self.padder.pad(signal, self.sample_rate*self.duration)
        signal = self.extractor.extract(signal)
    
        return signal

    def process(self, folder_path: str, list_names: list) -> np.ndarray:
        data = []
        for item in list_names:
            if os.path.exists(f"{folder_path}/{item}.flac"):
                signal = self.__process(f"{folder_path}/{item}.flac")
                data.append(signal)

        return np.array(data)