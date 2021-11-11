from collections import defaultdict
import glob
import os

import librosa
from torch.utils.data import Dataset

class MySTDataset(Dataset):
    """MyST dataset."""

    def __init__(self, data_path):
        """
        Args:
            data_path (str): path to MyST dataset.
        """
        self.data_path = data_path
        self.audio_files = glob.glob(os.path.join(self.data_path,
        										  '*/*/*/flac'))

    def __len__(self):
    	return len(self.audio_files)

    def __getitem__(self, idx):
    	audio_file = self.audio_files[idx]
    	audio_input, sample_rate = librosa.load(audio_file, sr=16000)
    	text_file = audio_file[:-4] + 'trn'
    	with open(text_file, 'r') as f:
    		text = f.read().lower()
    	return audio_input, sample_rate, text

class ZenodoDataset(Dataset):
	"""Zenodo dataset."""

	def __init__(self,
				 data_path,
				 native_extension='english_words_sentences/*_native/studio_mic/*/*.wav'):
		"""
		Args:
			data_path (str): path to Zenodo dataset.
		"""
		self.data_path = data_path
		self.audio_files = glob.glob(os.path.join(self.data_path, native_extension))

	def __len__(self):
		return len(self.audio_files)

	def __getitem__(self, idx):
		audio_file = self.audio_files[idx]
		audio_input, sample_rate = librosa.load(audio_file, sr=16000)
		# Get the file name and ignore the .wav extension.
		text = audio_file.split('/')[-1][:-4]
		# Split by underscore, join together by space, convert to lowercase.
		text = ' '.join(text.split('_')).lower()
		return audio_input, sample_rate, text

