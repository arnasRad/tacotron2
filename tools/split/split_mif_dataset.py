import os
from typing import List

import pandas as pd
from pandas import DataFrame


def shuffle(dataset: DataFrame) -> DataFrame:
    dataset = dataset[1:]
    dataset = dataset.sample(frac=1)
    return dataset[[0, 1]]
    # metadata.to_csv('metadata.csv', sep='|', header=None, index=None)


def add_input_dir_prefix(dataset: DataFrame, prefix: str) -> DataFrame:
    for idx, row in dataset.iterrows():
        row.iloc[0] = prefix + '/' + row.iloc[0]
    return dataset


def split_dataset(dataset: DataFrame, ratios: List[float]):
    train_set = dataset.sample(frac=ratios[0])
    dataset = dataset.drop(train_set.index)
    val_set = dataset.sample(frac=0.5)
    test_set = dataset.drop(val_set.index)
    return train_set, val_set, test_set


if __name__ == '__main__':
    split_ratios = [0.92, 0.5, 0.04]
    metadata = pd.read_csv('/media/arnas/SSD Disk/Speech Synthesis/Tacotron-2/datasets/mif-speech/metadata.csv',
                           sep='|', header=None)

    metadata = shuffle(metadata)
    metadata = add_input_dir_prefix(metadata, '/mnt/disks/deeplearning_disk/ino-voice/unzipped/mif-speech/audio')
    train_dataset, val_dataset, test_dataset = split_dataset(metadata, split_ratios)

    train_dataset.to_csv('train_filelist.txt', sep='|', header=False, index=False)
    val_dataset.to_csv('val_filelist.txt', sep='|', header=False, index=False)
    test_dataset.to_csv('test_filelist.txt', sep='|', header=False, index=False)
