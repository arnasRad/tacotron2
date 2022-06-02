from pathlib import Path

from torch.utils.data import DataLoader

from libs.data.dataset import TextMelCollate, ZipAudioWaveformDataset, SpectrogramDataset, split_to_train_and_val_sets


def prepare_dataloaders(hparams, train_and_validation_set_ratios=None):
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
    zip_data_files += read_files(Path(hparams.data_dir))

    datasets = []
    datasets += [create_zip_waveform_dataset(zip_data_file) for zip_data_file in zip_data_files]

    for i in range(len(datasets)):
        datasets[i].entry_metadata=[e for e in datasets[i].entry_metadata if 1.5 <= e.duration < 11.61]

    dataset = SpectrogramDataset(datasets, hparams)

    train_dataset, val_dataset = split_to_train_and_val_sets(dataset, train_and_validation_set_ratios)

    print("Training dataset size %s", len(train_dataset))

    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    train_loader = DataLoader(
        train_dataset,
        num_workers=1,
        shuffle=True,
        batch_size=hparams.batch_size,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_dataset, collate_fn
