import json
from io import BytesIO
from typing import List, Tuple, Iterable
from zipfile import ZipFile

import numpy as np
import torch
from pydub import AudioSegment
from scipy.io.wavfile import read
from torch.utils.data import Dataset, ConcatDataset, random_split

from libs.core import AudioRecordMetadata
from libs.model import layers
from libs.text import text_to_sequence


class TextMelCollate:
    """ Zero-pads model inputs and targets based on number of frames per step
    """

    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths


class SortableDataset(Dataset):
    def __getitem__(self, index):
        raise NotImplementedError

    def get_metadata(self, idx):
        raise NotImplementedError


class ZipAudioWaveformDataset(SortableDataset):
    def __init__(self, zip_filepath, ):
        self.zip_filepath = zip_filepath
        self.entry_metadata: List[AudioRecordMetadata] = list(self.read_manifest_metadata())

    def read_manifest_metadata(self):
        with ZipFile(str(self.zip_filepath)) as myzip:
            with myzip.open('manifest.jsona') as lines:
                for line in lines:
                    entry = json.loads(line)
                    yield AudioRecordMetadata(**entry)

    def __len__(self):
        return len(self.entry_metadata)

    def __getitem__(self, idx):
        metadata: AudioRecordMetadata = self.entry_metadata[idx]
        name, data = self.read_item(metadata.zip_entry_name)
        if name.endswith(".wav"):
            sample_rate, waveform = self.read_wav(data)
        elif name.endswith(".mp3"):
            waveform, sample_rate = self.read_mp3(data)
        else:
            raise ValueError(f"File extension for {name} was not recognized.")

        return waveform, sample_rate, metadata.text

    def get_metadata(self, idx) -> AudioRecordMetadata:
        return self.entry_metadata[idx]

    def read_wav(self, bytes):
        # return sf.read(BytesIO(bytes), dtype="float32", always_2d=True)
        return read(BytesIO(bytes))

    def read_mp3(self, bytes, normalized=False, channels=1):
        audio = AudioSegment.from_mp3(BytesIO(bytes))
        audio = audio.set_channels(channels)

        y = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if audio.channels == 2:
            y = y.reshape((-1, 2))
        if normalized:
            return y / 2 ** 15, audio.frame_rate, audio.duration_seconds
        else:
            return y, audio.frame_rate, audio.duration_seconds

    def read_item(self, name) -> Tuple[str, bytes]:
        with ZipFile(self.zip_filepath) as myzip:
            with myzip.open(name) as myfile:
                wav = myfile.read()
                return name, wav


class SpectrogramDataset(ZipAudioWaveformDataset, ConcatDataset):
    def __init__(self, datasets: Iterable[SortableDataset], hparams):
        ConcatDataset.__init__(self, datasets)
        self.max_wav_value = hparams.max_wav_value
        self.text_cleaners = hparams.text_cleaners
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

    def __getitem__(self, index):
        waveform, sample_rate, text = ConcatDataset.__getitem__(self, index)
        return self.get_mel_text_pair(waveform, sample_rate, text)

    def __len__(self):
        return ConcatDataset.__len__(self)

    def get_mel_text_pair(self, waveform, sample_rate, text):
        text = self._get_text(text)
        mel = self._get_mel(waveform, sample_rate)
        return text, mel

    def _get_mel(self, waveform, sample_rate):
        if sample_rate != self.stft.sampling_rate:
            raise ValueError(
                f"Sample rate ({sample_rate}) doesn't match target sample rate ({self.stft.sampling_rate})")

        audio = torch.FloatTensor(waveform.astype(np.float32))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)

        return melspec

    def _get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm


def calculate_train_and_val_set_sizes(total_size: int, ratios: List[float]):
    if ratios is None:
        ratios = [0.95, 0.05]
    else:
        assert len(ratios) == 2 and sum(ratios) == 1

    train_size = round(total_size * ratios[0])
    val_size = total_size - train_size

    return train_size, val_size


def split_to_train_and_val_sets(dataset: SpectrogramDataset, ratios: List[float]):
    total_size = len(dataset)
    print("Total dataset size %s", total_size)
    train_size, val_size = calculate_train_and_val_set_sizes(total_size, ratios)

    return random_split(dataset, [train_size, val_size])
