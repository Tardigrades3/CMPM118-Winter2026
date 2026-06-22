from . import signal
import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

def padding(batch):
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    lengths = torch.tensor([len(seq) for seq in sequences])

    sequences = [torch.tensor(seq, dtype=torch.float32) if not isinstance(seq, torch.Tensor) else seq for seq in sequences]
    labels = torch.tensor(labels, dtype=torch.long)

    padded_batch = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    batch_size, max_len, _ = padded_batch.shape
    attention_mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
    attention_mask = attention_mask.long()

    return padded_batch, labels, attention_mask


def build_ss_task_streams(exercise_number, path, shuffle, batch_size=32, num_subjects=27):
    """Domain-Incremental Learning (DIL) stream: loops through subjects for a specific exercise."""
    task_stream = []

    for subject in range(num_subjects):
        base_path = f"{path}/s{subject + 1}/S{subject + 1}_A1_E{exercise_number}.mat"
        data = signal.load_data(base_path)

        x_train, y_train, x_test, y_test, _cw = signal.preprocessing_internals(data)

        train_loader = DataLoader(
            signal.NinaProDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=padding,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        test_loader = DataLoader(
            signal.NinaProDataset(x_test, y_test),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=padding
        )

        task_stream.append({
            'task_id': f'subject_{subject+1}',
            'train': train_loader,
            'test': test_loader
        })

    return task_stream


def build_cil_multi_exercise_stream(subject_id, path, batch_size=32, shuffle=True):
    """Class-Incremental Learning (CIL) stream: loops through 3 exercises for a single subject."""
    exercises = [1, 2, 3]
    task_streams = []

    current_class_offset = 0
    max_label_in_dataset = 0

    for ex_num in exercises:
        print(f"Loading Subject {subject_id}, Exercise {ex_num} for CIL...")

        base_path = f"{path}/s{subject_id}/S{subject_id}_A1_E{ex_num}.mat"
        data = signal.load_data(base_path)
        x_train, y_train, x_test, y_test, _cw = signal.preprocessing_internals(data)

        # Class 0 (Rest) is shared; all other classes are offset to make labels disjoint.
        y_train_offset = np.where(y_train == 0, 0, y_train + current_class_offset)
        y_test_offset = np.where(y_test == 0, 0, y_test + current_class_offset)

        if len(y_train_offset) > 0:
            max_label_in_dataset = max(max_label_in_dataset, np.max(y_train_offset))

        train_loader = DataLoader(
            signal.NinaProDataset(x_train, y_train_offset),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=padding,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        test_loader = DataLoader(
            signal.NinaProDataset(x_test, y_test_offset),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=padding
        )

        task_streams.append({
            'task_id': f"Subject_{subject_id}_Exercise_{ex_num}",
            'train': train_loader,
            'test': test_loader
        })

        if len(y_train) > 0:
            current_class_offset += np.max(y_train)

    return task_streams, int(max_label_in_dataset) + 1
