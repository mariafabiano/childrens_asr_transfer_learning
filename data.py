from collections import defaultdict
import glob
import os
import re

import librosa
from torch.utils.data import Dataset

from collections import defaultdict
import glob
import os
import re
import numpy as np

import librosa
from torch.utils.data import Dataset

class MySTDataset(Dataset):
    """MyST dataset."""

    def __init__(self, data_path, sample_rate=16000,
		             chars_to_ignore_regex='[\,\?\.\!\-\;\:\"\<\>\_]'):
        """
        Args:
            data_path (str): path to MyST dataset.
        """
        self.data_path = data_path
        self.audio_files = glob.glob(os.path.join(self.data_path,
        										  '*/*/*.flac'))
        self.sample_rate = sample_rate
        self.remove_short_audio()
        print(f'# of audio files after removing short audio: {len(self.audio_files)}')
        self.chars_to_ignore = chars_to_ignore_regex
        self.processor = None

    def init_processor(self, processor):
      self.processor = processor

    def extract_all_chars(self):
        vocab = set()
        for audio_file in self.audio_files:
            text_file = audio_file[:-4] + 'trn'
            text = self.read_text_file(text_file)
            vocab.update(text)
        return {"vocab": [vocab]}

    def remove_short_audio(self):
      min_input_length_in_sec = 1.0
      files_to_keep = []
      for i in range(len(self.audio_files)):
         audio_input, sample_rate = librosa.load(self.audio_files[i], sr=self.sample_rate)
         if len(audio_input) >= sample_rate*1:
           files_to_keep.append(self.audio_files[i])
      self.audio_files = files_to_keep
       
    def prepare_dataset(self, audio_array, text):
        batch = {}

        # batched output is "un-batched" to ensure mapping is correct
        batch["input_values"] = self.processor(np.array(audio_array), sampling_rate=self.sample_rate).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        
        with self.processor.as_target_processor():
            batch["labels"] = self.processor(text).input_ids
        return batch

    def read_text_file(self, text_file):
        with open(text_file, 'r') as f:
            text = f.read().lower().strip()
            text = re.sub(self.chars_to_ignore, '', text) + " "
            text = text.replace('(())', 's') # Ignore noise.
        return text

    def __len__(self):
        return len(self.audio_files)
		
    def __getitem__(self, idx):
        if isinstance(idx, int):
            audio_file = self.audio_files[idx]
            audio_input, sample_rate = librosa.load(audio_file, sr=self.sample_rate)
            text_file = audio_file[:-4] + 'trn'
            text = self.read_text_file(text_file)
            if self.processor is not None:
                prepared_audio_dict = self.prepare_dataset(audio_input, text)
                return prepared_audio_dict
            return {'audio': {'array': audio_input, 'path': audio_file, 'sampling_rate': self.sample_rate}, 'file': audio_file, 'text': text}

        else:
            audio_files = None
            if isinstance(idx, slice):
                audio_files = self.audio_files[idx]
            elif isinstance(idx, list):
                audio_files = [self.audio_files[i] for i in idx]
            audio_input = [librosa.load(audio_file, sr=self.sample_rate)[0] for audio_file in audio_files]
            text_files = [x[:-4] + 'trn' for x in audio_files]
            texts = []
            for text_file in text_files:
                text = self.read_text_file(text_file)
                texts.append(text)
            return {'audio': [{'array': audio, 'path': path, 'sampling_rate': self.sample_rate} for audio, path in zip(audio_input, audio_files)], 'file': audio_files, 'text': texts}


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
		text = ' '.join(text.split('_')).lower().strip()
		return {'audio': audio_input, 'text': text, 'file': audio_file}

