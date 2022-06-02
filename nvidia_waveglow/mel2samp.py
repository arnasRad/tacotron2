# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************\
import argparse
import json
import os
import random
import sys
from io import BytesIO
from pathlib import Path
from typing import List, Tuple
from zipfile import ZipFile

import torch
import torch.utils.data
from scipy.io.wavfile import read
# We're using the audio processing from TacoTron2 to make sure it matches
from torch.utils.data import ConcatDataset

sys.path.insert(0, 'tacotron2')
from libs.model.layers import TacotronSTFT
from libs.data.dataset import SortableDataset
from libs.core import AudioRecordMetadata

MAX_WAV_VALUE = 32768.0


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate


def load_bytes_to_torch(wav_bytes):
    sampling_rate, data = read(BytesIO(wav_bytes))
    return torch.from_numpy(data).float(), sampling_rate


def prepare_dataset(dataset_zips_directory, segment_length, filter_length,
                    hop_length, win_length, sampling_rate, mel_fmin, mel_fmax):
    def read_files(path):
        data_files = list(path.glob('**/*.zip'))
        data_files = filter(Path.is_file, data_files)
        return list(data_files)

    def create_zip_waveform_dataset(file):
        print("Creating dataset for %s", file)
        dataset = ZipAudioWaveformDataset(
            zip_filepath=file,
        )
        return dataset

    zip_data_files = []
    zip_data_files += read_files(Path(dataset_zips_directory))

    datasets = []
    datasets += [create_zip_waveform_dataset(zip_data_file) for zip_data_file in zip_data_files]
    for i in range(len(datasets)):
        datasets[i].entry_metadata = [e for e in datasets[i].entry_metadata if 1.5 <= e.duration < 11.61]

    return Mel2Samp(datasets, segment_length, filter_length,
                    hop_length, win_length, sampling_rate, mel_fmin, mel_fmax)


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
            waveform, sample_rate = load_bytes_to_torch(data)
        else:
            raise ValueError(f"File extension for {name} was not recognized.")

        return waveform, sample_rate

    def get_metadata(self, idx) -> AudioRecordMetadata:
        return self.entry_metadata[idx]

    def read_item(self, name) -> Tuple[str, bytes]:
        with ZipFile(self.zip_filepath) as myzip:
            with myzip.open(name) as myfile:
                wav = myfile.read()
                return name, wav


class Mel2Samp(ZipAudioWaveformDataset, ConcatDataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, datasets, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax):
        ConcatDataset.__init__(self, datasets)
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate

    def get_mel(self, audio):
        audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, index):
        # Read audio
        audio, sampling_rate = ConcatDataset.__getitem__(self, index)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start + self.segment_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data

        mel = self.get_mel(audio)
        audio = audio / MAX_WAV_VALUE

        return (mel, audio)

    def __len__(self):
        return ConcatDataset.__len__(self)


# ===================================================================
# Takes directory of clean audio and makes directory of spectrograms
# Useful for making test sets
# ===================================================================
if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='Output directory')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    data_config = json.loads(data)["data_config"]
    mel2samp = Mel2Samp(**data_config)

    filepaths = files_to_list(args.filelist_path)

    # Make directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)

    for filepath in filepaths:
        audio, sr = load_wav_to_torch(filepath)
        melspectrogram = mel2samp.get_mel(audio)
        filename = os.path.basename(filepath)
        new_filepath = args.output_dir + '/' + filename + '.pt'
        print(new_filepath)
        torch.save(melspectrogram, new_filepath)
